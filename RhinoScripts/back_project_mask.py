import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
import System
import os
import json
import numpy as np
import math
from System.Drawing import Bitmap, Color, Rectangle
from System.Drawing.Imaging import PixelFormat
from System.Runtime.InteropServices import Marshal

def extract_mask_data(mask_bitmap):
    """Extract binary mask data"""
    rect = Rectangle(0, 0, mask_bitmap.Width, mask_bitmap.Height)
    bitmap_data = mask_bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, 
                                      PixelFormat.Format32bppArgb)
    
    try:
        # Get pixel data
        byte_count = bitmap_data.Stride * bitmap_data.Height
        byte_array = System.Array.CreateInstance(System.Byte, byte_count)
        Marshal.Copy(bitmap_data.Scan0, byte_array, 0, byte_count)
        
        # Convert to numpy array
        labels = np.zeros((mask_bitmap.Height, mask_bitmap.Width), dtype=np.uint8)
        
        # Process each pixel - just check if it's white (non-black)
        for y in range(mask_bitmap.Height):
            for x in range(mask_bitmap.Width):
                offset = y * bitmap_data.Stride + x * 4
                b = byte_array[offset]
                g = byte_array[offset + 1]
                r = byte_array[offset + 2]
                
                # If any channel has significant value, consider it white
                if r > 127 or g > 127 or b > 127:
                    labels[y, x] = 255
    
    finally:
        mask_bitmap.UnlockBits(bitmap_data)
    
    # Print stats about the mask
    nonzero = np.count_nonzero(labels)
    total = labels.size
    percent = (nonzero / total) * 100
    print(f"Mask stats: {nonzero} white pixels out of {total} ({percent:.2f}%)")
    
    return labels


def back_project_using_rays(mask_filepath, depth_filepath, view_info, mesh_guid, 
                           debug_rays=False, fov_adjustment=1.0, max_sample_points=60,
                           ray_length_multiplier=10.0):
    """Back-project mask points using ray casting with limited sample points"""
    # Load mask image
    mask_bitmap = Bitmap(mask_filepath)
    mask_width = mask_bitmap.Width
    mask_height = mask_bitmap.Height
    
    # Get camera information
    camera_position = np.array(view_info["camera_position"])
    camera_target = np.array(view_info["target"])
    
    # Calculate camera basis vectors
    camera_forward = camera_target - camera_position
    camera_distance = np.linalg.norm(camera_forward)
    camera_forward = camera_forward / camera_distance  # Normalize
    
    # Create camera coordinate system
    world_up = np.array([0, 0, 1])  # Z-up in Rhino
    camera_right = np.cross(camera_forward, world_up)
    if np.linalg.norm(camera_right) < 0.001:
        world_up = np.array([0, 1, 0])  # Use Y as alternative
        camera_right = np.cross(camera_forward, world_up)
    camera_right = camera_right / np.linalg.norm(camera_right)  # Normalize
    camera_up = np.cross(camera_right, camera_forward)
    camera_up = camera_up / np.linalg.norm(camera_up)  # Normalize
    
    # Get viewport info
    is_perspective = view_info["projection_info"]["is_perspective"]
    print(f"Projection type: {'Perspective' if is_perspective else 'Orthographic'}")
    
    # Extract mask data (white pixels only)
    mask_data = extract_mask_data(mask_bitmap)
    
    # Get mesh for ray casting
    mesh = rs.coercemesh(mesh_guid)
    if not mesh:
        print("Error: Could not coerce mesh for ray casting")
        return np.array([])
    
    # Get mesh bounding box for debugging
    mesh_bbox = mesh.GetBoundingBox(True)
    bbox_min = mesh_bbox.Min
    bbox_max = mesh_bbox.Max
    print(f"Mesh bounding box: Min({bbox_min.X:.2f}, {bbox_min.Y:.2f}, {bbox_min.Z:.2f}), Max({bbox_max.X:.2f}, {bbox_max.Y:.2f}, {bbox_max.Z:.2f})")
    
    # Create a layer for debug rays if needed
    debug_layer = "DebugRays"
    if debug_rays and not rs.IsLayer(debug_layer):
        rs.AddLayer(debug_layer)
        rs.LayerColor(debug_layer, rs.CreateColor(0, 255, 255))  # Cyan
    
    # Get FOV information from the enhanced data
    if "fov" in view_info and isinstance(view_info["fov"], dict):
        v_fov = view_info["fov"]["vertical"]
        h_fov = view_info["fov"]["horizontal"]
    else:
        # Fallback to estimated FOV
        aspect_ratio = float(mask_width) / float(mask_height)
        v_fov = 60.0  # Default if not available
        h_fov = v_fov * aspect_ratio
    
    # Apply FOV adjustment
    v_fov *= fov_adjustment
    h_fov *= fov_adjustment
    print(f"Using FOV: vertical={v_fov:.2f}, horizontal={h_fov:.2f}")
    
    # Calculate ray offsets based on adjusted FOV
    h_tan = math.tan(math.radians(h_fov / 2))
    v_tan = math.tan(math.radians(v_fov / 2))
    
    # Find white pixels for sampling
    white_pixels = []
    for y in range(mask_height):
        for x in range(mask_width):
            if mask_data[y, x] > 0:  # If white
                white_pixels.append((x, y))
    
    # Limit to max_sample_points by selecting evenly distributed samples
    sample_pixels = []
    if len(white_pixels) > 0:
        if len(white_pixels) <= max_sample_points:
            sample_pixels = white_pixels
        else:
            # Select evenly spaced samples
            step = len(white_pixels) // max_sample_points
            sample_pixels = [white_pixels[i] for i in range(0, len(white_pixels), step)]
            # Ensure we don't exceed max_sample_points
            sample_pixels = sample_pixels[:max_sample_points]
    
    print(f"Selected {len(sample_pixels)} sample points out of {len(white_pixels)} white pixels")
    
    # Calculate max ray length for testing (longer rays)
    mesh_size = max(
        bbox_max.X - bbox_min.X,
        bbox_max.Y - bbox_min.Y,
        bbox_max.Z - bbox_min.Z
    )
    max_ray_length = max(camera_distance, mesh_size) * ray_length_multiplier
    
    # Keep track of previous layer
    previous_layer = rs.CurrentLayer()
    hits = 0
    misses = 0
    
    # Array to store results
    points_3d = []
    
    # Process only the selected sample points
    for x, y in sample_pixels:
        # Calculate normalized device coordinates (-1 to 1)
        ndc_x = (x / float(mask_width - 1)) * 2.0 - 1.0
        ndc_y = 1.0 - (y / float(mask_height - 1)) * 2.0  # Y flipped
        
        # Calculate ray direction based on projection type
        if is_perspective:
            # For perspective projection with adjusted FOV
            ray_dir_x = ndc_x * h_tan
            ray_dir_y = ndc_y * v_tan
            ray_dir_z = 1.0  # Forward
            
            # Create normalized ray direction in view space
            ray_dir_view = np.array([ray_dir_x, ray_dir_y, ray_dir_z])
            ray_dir_view = ray_dir_view / np.linalg.norm(ray_dir_view)
            
            # Transform to world space using camera basis vectors
            ray_dir_world = (
                ray_dir_view[0] * camera_right + 
                ray_dir_view[1] * camera_up + 
                ray_dir_view[2] * camera_forward
            )
        else:
            # For orthographic projection with adjusted FOV
            view_width = camera_distance * h_tan * 2
            view_height = camera_distance * v_tan * 2
            
            # Calculate offset from center
            offset_x = ndc_x * (view_width / 2)
            offset_y = ndc_y * (view_height / 2)
            
            # Ray starts at camera plane, offset by x,y
            ray_start = (
                camera_position + 
                camera_right * offset_x + 
                camera_up * offset_y
            )
            
            # Direction is parallel to forward
            ray_dir_world = camera_forward
        
        # Create Rhino ray for intersection
        ray_origin = Rhino.Geometry.Point3d(
            camera_position[0], camera_position[1], camera_position[2]
        )
        ray_dir = Rhino.Geometry.Vector3d(
            ray_dir_world[0], ray_dir_world[1], ray_dir_world[2]
        )
        
        # Use a much longer ray for testing - KEY FIX #1
        ray_dir.Unitize()  # Ensure unit vector
        ray = Rhino.Geometry.Ray3d(ray_origin, ray_dir)
        
        # Create a line going through the mesh for intersection - KEY FIX #2
        # Sometimes mesh.ray doesn't work well, but Line.MeshIntersection does
        line_end = ray_origin + ray_dir * max_ray_length
        line = Rhino.Geometry.Line(ray_origin, line_end)
        
        # Try multiple intersection methods - KEY FIX #3
        intersection_found = False
        hit_point = None
        
        # Method 1: MeshRay
        intersection_param = Rhino.Geometry.Intersect.Intersection.MeshRay(mesh, ray)
        if intersection_param >= 0:
            intersection_found = True
            hit_point = ray_origin + ray_dir * intersection_param
        
        # Method 2: Line-Mesh intersection if ray casting failed
        if not intersection_found:
            intersections = Rhino.Geometry.Intersect.Intersection.MeshLine(mesh, line)
            if intersections and len(intersections) > 0:
                intersection_found = True
                # Use the closest intersection point
                closest_t = float('inf')
                for intersection in intersections:
                    if intersection.ParameterA < closest_t:
                        closest_t = intersection.ParameterA
                        hit_point = line.PointAt(closest_t)
        
        if intersection_found:  # Hit
            hits += 1
            
            # Convert to numpy array for storage
            hit_point_array = np.array([hit_point.X, hit_point.Y, hit_point.Z])
            points_3d.append(hit_point_array)
            
            # Draw debug rays for sample points
            if debug_rays:
                rs.CurrentLayer(debug_layer)
                line = rs.AddLine(
                    [ray_origin.X, ray_origin.Y, ray_origin.Z],
                    [hit_point.X, hit_point.Y, hit_point.Z]
                )
                rs.ObjectColor(line, rs.CreateColor(0, 255, 0))  # Green for hits
        else:
            misses += 1
            
            # Draw debug rays for misses
            if debug_rays:
                rs.CurrentLayer(debug_layer)
                # Draw a limited length ray for misses
                end_point = ray_origin + ray_dir * max_ray_length
                line = rs.AddLine(
                    [ray_origin.X, ray_origin.Y, ray_origin.Z],
                    [end_point.X, end_point.Y, end_point.Z]
                )
                rs.ObjectColor(line, rs.CreateColor(255, 0, 0))  # Red for misses
    
    # Restore previous layer
    rs.CurrentLayer(previous_layer)
    
    print(f"Ray statistics: {hits} hits, {misses} misses")
    
    # If no hits, draw camera and target points for debugging
    if hits == 0 and debug_rays:
        rs.CurrentLayer(debug_layer)
        # Draw camera position
        cam_point = rs.AddPoint(camera_position[0], camera_position[1], camera_position[2])
        rs.ObjectColor(cam_point, rs.CreateColor(255, 255, 0))  # Yellow for camera
        
        # Draw target position
        target_point = rs.AddPoint(camera_target[0], camera_target[1], camera_target[2])
        rs.ObjectColor(target_point, rs.CreateColor(0, 255, 255))  # Cyan for target
        
        # Draw a line from camera to target
        cam_line = rs.AddLine(
            [camera_position[0], camera_position[1], camera_position[2]],
            [camera_target[0], camera_target[1], camera_target[2]]
        )
        rs.ObjectColor(cam_line, rs.CreateColor(255, 0, 255))  # Magenta for camera line
    
    return np.array(points_3d)


def process_all_views(output_folder="C:\\Screenshots", mesh_guid=None, 
                     debug_rays=False, fov_adjustment=1.0, max_sample_points=60):
    """Process all views in the camera data"""
    # Load camera data
    camera_data_path = os.path.join(output_folder, "camera_data.json")
    if not os.path.exists(camera_data_path):
        print(f"Camera data not found at {camera_data_path}")
        return
        
    with open(camera_data_path, 'r') as json_file:
        camera_data = json.load(json_file)
    
    # Process each view
    results = {}
    for view_info in camera_data["views"]:
        target_filename = view_info["filename"]
        
        # Create a layer for visualization
        base_layer = f"RayCast_{target_filename.replace('.jpeg', '')}"
        if not rs.IsLayer(base_layer):
            rs.AddLayer(base_layer)
            rs.LayerColor(base_layer, rs.CreateColor(255, 0, 0))  # Red for visibility
        else:
            # Clear existing objects from this layer
            rs.CurrentLayer(base_layer)
            objects = rs.ObjectsByLayer(base_layer)
            if objects:
                rs.DeleteObjects(objects)
        
        # Get mask filename
        mask_filename = target_filename.replace(".jpeg", "_combined_mask.png")
        mask_filepath = os.path.join(output_folder, mask_filename)
        
        # Check if mask exists
        if not os.path.exists(mask_filepath):
            print(f"Mask not found: {mask_filepath}, trying alternatives...")
            
            # Try alternative mask names
            alt_mask = target_filename.replace(".jpeg", "_mask.png")
            alt_mask_path = os.path.join(output_folder, alt_mask)
            if os.path.exists(alt_mask_path):
                mask_filepath = alt_mask_path
                print(f"Using alternative mask: {alt_mask}")
            else:
                print(f"No suitable mask found for {target_filename}, skipping this view")
                continue
        
        # Get depth data path
        depth_filepath = os.path.join(output_folder, view_info["depth_npy_filename"])
        if not os.path.exists(depth_filepath):
            print(f"Depth data not found: {depth_filepath}, checking alternatives...")
            
            # Try to find alternative depth data
            alt_depth = target_filename.replace(".jpeg", "_depth.npy")
            alt_depth_path = os.path.join(output_folder, alt_depth)
            if os.path.exists(alt_depth_path):
                depth_filepath = alt_depth_path
                print(f"Using alternative depth data: {alt_depth}")
            else:
                print(f"No depth data found for {target_filename}, continuing without depth")
        
        print(f"\n--- Processing view: {target_filename} ---")
        
        # Back-project points using ray casting with limited sample points
        points_3d = back_project_using_rays(
            mask_filepath, depth_filepath, view_info, mesh_guid, 
            debug_rays, fov_adjustment, max_sample_points, 
            ray_length_multiplier=10.0  # Use longer rays to ensure they hit the mesh
        )
        
        if len(points_3d) == 0:
            print(f"No valid points were back-projected for view {target_filename}")
            continue
        
        # Add points to Rhino
        previous_layer = rs.CurrentLayer()
        rs.CurrentLayer(base_layer)
        
        points_objs = []
        for point in points_3d:
            point_obj = rs.AddPoint([point[0], point[1], point[2]])
            points_objs.append(point_obj)
        
        # Create a group for the points
        if points_objs:
            group_name = f"BackProjected_{target_filename.replace('.jpeg', '')}"
            rs.AddGroup(group_name)
            rs.AddObjectsToGroup(points_objs, group_name)
        
        # Restore previous layer
        rs.CurrentLayer(previous_layer)
        
        print(f"Created {len(points_3d)} points for view {target_filename}")
        
        # Store the results
        results[target_filename] = points_3d
    
    return results


def apply_colors_to_mesh(mesh_guid, points_3d, color=None):
    """Apply a color to mesh faces that are closest to the back-projected points"""
    if color is None:
        color = rs.CreateColor(255, 0, 0)  # Default to red
    
    # Get the mesh
    mesh = rs.coercemesh(mesh_guid)
    if not mesh:
        print(f"Could not coerce {mesh_guid} to a mesh.")
        return None
    
    # Dictionary to store face indices and their assigned labels
    face_labels = {}
    
    # For each 3D point
    for i, point in enumerate(points_3d):
        # Find closest point on mesh
        result = rs.MeshClosestPoint(mesh_guid, [point[0], point[1], point[2]])
        
        if result:
            closest_point, face_index = result
            
            # If we found a valid face
            if face_index >= 0:
                # Calculate distance
                distance = rs.Distance(point, closest_point)
                
                # If this face doesn't have a label yet, or we're closer than previous point
                if face_index not in face_labels:
                    face_labels[face_index] = (255, distance)
                else:
                    if distance < face_labels[face_index][1]:
                        face_labels[face_index] = (255, distance)
    
    # Clean up the dictionary to only contain the label, not the distance
    labeled_mesh = {face: label_data[0] for face, label_data in face_labels.items()}
    
    # Apply colors to the mesh
    try:
        # Create a new colored mesh
        colored_mesh = Rhino.Geometry.Mesh()
        colored_mesh.CopyFrom(mesh)
        
        # Enable vertex colors
        colored_mesh.VertexColors.CreateMonotoneMesh(Color.LightGray)
        
        # Default gray for unlabeled faces
        default_color = Color.LightGray
        
        # Apply colors by face
        for face_idx in range(colored_mesh.Faces.Count):
            # Determine color for this face
            if face_idx in labeled_mesh:
                # Use the specified color
                face_color = Color.FromArgb(color.R, color.G, color.B)
            else:
                face_color = default_color
            
            # Get the face
            face = colored_mesh.Faces[face_idx]
            
            # Set vertex colors for this face
            colored_mesh.VertexColors[face.A] = face_color
            colored_mesh.VertexColors[face.B] = face_color
            colored_mesh.VertexColors[face.C] = face_color
            if face.IsQuad:
                colored_mesh.VertexColors[face.D] = face_color
        
        # Add the colored mesh to the document
        new_mesh_id = sc.doc.Objects.AddMesh(colored_mesh)
        
        if new_mesh_id != System.Guid.Empty:
            print(f"Successfully created colored mesh with ID: {new_mesh_id}")
            sc.doc.Views.Redraw()
        else:
            print("Failed to add colored mesh to document")
        
    except Exception as e:
        print(f"Error applying colors to mesh: {str(e)}")
    
    return labeled_mesh


# Main execution
if __name__ == "__main__":
    # Fixed output folder path
    output_folder = "C:\\Screenshots"
    
    # Verify folder exists
    if not os.path.exists(output_folder):
        print(f"Warning: Output folder {output_folder} does not exist!")
        output_folder = rs.GetString("Path to data folder with masks and depth data", output_folder)
        if not os.path.exists(output_folder):
            print("Invalid folder path")
            exit()
    
    # Ask user to select a mesh for ray casting
    mesh_guid = rs.GetObject("Select mesh for ray casting", 32)  # 32 is mesh filter
    if not mesh_guid:
        print("Error: No mesh selected. Ray casting requires a mesh.")
        exit()
    
    # Ask for FOV adjustment factor
    fov_adjustment = rs.GetReal("FOV adjustment factor (1.0=original, 0.5=narrower, 2.0=wider)", 1.0, 0.1, 5.0)
    if fov_adjustment is None:
        fov_adjustment = 1.0  # Use original FOV by default
    
    # Ask if user wants to debug with visual rays (warning: this can create a lot of geometry)
    debug_rays = rs.MessageBox("Draw debug rays to visualize sample rays?\nWarning: This can create a lot of geometry with multiple views.", 4) == 6  # 6 is Yes
    
    # Ask for number of sample points
    max_sample_points = rs.GetInteger("Maximum number of sample points per view", 100, 10, 1000)
    if max_sample_points is None:
        max_sample_points = 100  # Default
    
    # Process all views
    print(f"Processing all views with maximum {max_sample_points} sample points per view")
    results = process_all_views(
        output_folder=output_folder,
        mesh_guid=mesh_guid,
        debug_rays=debug_rays,
        fov_adjustment=fov_adjustment,
        max_sample_points=max_sample_points
    )
    
    # Ask if user wants to color the mesh using points from all views
    if results and len(results) > 0:
        if rs.MessageBox("Apply colors to mesh based on back-projected points from all views?", 4) == 6:
            # Combine all points from all views
            all_points = np.vstack([points for points in results.values()])
            apply_colors_to_mesh(mesh_guid, all_points)