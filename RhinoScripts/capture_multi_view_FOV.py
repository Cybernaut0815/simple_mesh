import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
import System
import math
import os
import time
import json
import numpy as np
from System.Drawing import Bitmap, Rectangle
from System.Drawing.Imaging import PixelFormat, ImageFormat, BitmapData
from System.Runtime.InteropServices import GCHandle, GCHandleType, Marshal

def capture_multi_view_screenshots():
    # Configuration - all parameters collected at the beginning
    output_folder = rs.GetString("Output folder path", "C:\\Screenshots", "Folder to save screenshots")
    number_of_views = rs.GetInteger("Number of views", 8, 2, 36)
    elevation_angles = [30, 0, -30]  # Degrees above/below horizon
    image_width = rs.GetInteger("Image width (pixels)", 1024, 256, 4096)
    image_height = rs.GetInteger("Image height (pixels)", 1024, 256, 4096)
    
    # Add zoom factor parameter - asked only once
    zoom_factor = rs.GetReal("Zoom factor (1.0 = standard, >1.0 = more zoomed out)", 1.5, 0.5, 5.0)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get objects to focus on - ensure this always happens
    objects = rs.GetObjects("Select objects to capture", 0, preselect=False, select=True)
    if not objects:
        print("No objects selected. Cancelling operation.")
        return
    
    # Get bounding box of all objects
    bbox_min = [float('inf'), float('inf'), float('inf')]
    bbox_max = [float('-inf'), float('-inf'), float('-inf')]
    
    for obj in objects:
        bbox = rs.BoundingBox(obj)
        if bbox:
            for point in bbox:
                for i in range(3):
                    bbox_min[i] = min(bbox_min[i], point[i])
                    bbox_max[i] = max(bbox_max[i], point[i])
    
    # Calculate center point and radius
    center = [(bbox_min[0] + bbox_max[0]) / 2,
              (bbox_min[1] + bbox_max[1]) / 2,
              (bbox_min[2] + bbox_max[2]) / 2]
    
    dimensions = [bbox_max[0] - bbox_min[0],
                  bbox_max[1] - bbox_min[1],
                  bbox_max[2] - bbox_min[2]]
    
    # Apply zoom factor directly to the radius calculation
    radius = max(dimensions) * 3.0 * zoom_factor
    
    # Store initial view
    initial_view = sc.doc.Views.ActiveView.ActiveViewport.Name
    
    # Store which objects were selected so we can restore selection later if needed
    selected_objects = objects.copy() if isinstance(objects, list) else list(objects)
    
    # Deselect all objects before taking screenshots
    rs.UnselectAllObjects()
    
    # Create a dictionary to store camera positions and parameters
    camera_data = {
        "target": center,
        "image_width": image_width,
        "image_height": image_height,
        "zoom_factor": zoom_factor,
        "object_dimensions": dimensions,
        "bounding_box": {
            "min": bbox_min,
            "max": bbox_max
        },
        "views": []
    }
    
    # Capture screenshots at different angles
    image_count = 0
    
    for elevation in elevation_angles:
        for i in range(number_of_views):
            # Calculate camera position
            angle = (i * 360.0) / number_of_views
            angle_rad = math.radians(angle)
            elevation_rad = math.radians(elevation)
            
            # Convert spherical to Cartesian coordinates
            x = center[0] + radius * math.cos(elevation_rad) * math.cos(angle_rad)
            y = center[1] + radius * math.cos(elevation_rad) * math.sin(angle_rad)
            z = center[2] + radius * math.sin(elevation_rad)
            
            camera_location = [x, y, z]
            
            # Get active view
            view = sc.doc.Views.ActiveView
            view_name = view.ActiveViewport.Name
            
            # Set the camera position and target
            rs.ViewCameraTarget(view_name, camera_location, center)
            
            # Short delay to ensure the view updates
            time.sleep(0.2)
            
            # Make sure objects are deselected before each screenshot
            rs.UnselectAllObjects()
            
            # Update the view
            view.Redraw()
            
            # Create filename
            filename = f"view_elev{elevation}_angle{angle:.0f}.jpeg"
            file_path = os.path.join(output_folder, filename)
            
            # Create filename for depth image
            depth_filename = f"view_elev{elevation}_angle{angle:.0f}_depth.png"
            depth_file_path = os.path.join(output_folder, depth_filename)
            
            # Get current view and viewport for projections
            viewport = view.ActiveViewport
            
            # Get view frustum and projection information
            frustum = viewport.GetFrustum()
            near_dist = frustum[0]
            far_dist = frustum[1]
            
            # Get FOV (Field of View) information
            # Calculate vertical FOV in degrees - this is what we need to add back
            camera_lens_angle = viewport.Camera35mmLensLength
            vertical_fov = 2 * math.degrees(math.atan(18.0 / camera_lens_angle))
            
            # Calculate horizontal FOV based on aspect ratio
            aspect_ratio = float(image_width) / float(image_height)
            horizontal_fov = vertical_fov
            if aspect_ratio > 1.0:
                horizontal_fov = vertical_fov * aspect_ratio
            
            # Get view projection and camera transformations
            view_proj = viewport.GetTransform(Rhino.DocObjects.CoordinateSystem.World, Rhino.DocObjects.CoordinateSystem.Screen)
            cam_to_world = viewport.GetTransform(Rhino.DocObjects.CoordinateSystem.Camera, Rhino.DocObjects.CoordinateSystem.World)
            world_to_cam = viewport.GetTransform(Rhino.DocObjects.CoordinateSystem.World, Rhino.DocObjects.CoordinateSystem.Camera)
            
            # Calculate view to screen transform
            # Since there's no direct View coordinate system, we'll create it manually
            # This is typically the world to screen transform
            view_to_screen = viewport.GetTransform(Rhino.DocObjects.CoordinateSystem.World, Rhino.DocObjects.CoordinateSystem.Screen)
            
            # Extract camera-x, camera-y, camera-z vectors
            camera_x = [cam_to_world.M00, cam_to_world.M10, cam_to_world.M20]
            camera_y = [cam_to_world.M01, cam_to_world.M11, cam_to_world.M21]
            camera_z = [cam_to_world.M02, cam_to_world.M12, cam_to_world.M22]
            
            # Store camera information for this view
            view_data = {
                "filename": filename,
                "depth_filename": depth_filename,
                "camera_position": camera_location,
                "target": center,
                "angle_degrees": angle,
                "elevation_degrees": elevation,
                "camera_direction": [
                    center[0] - camera_location[0],
                    center[1] - camera_location[1], 
                    center[2] - camera_location[2]
                ],
                "near_plane": near_dist,
                "far_plane": far_dist,
                "fov": {
                    "vertical": vertical_fov,
                    "horizontal": horizontal_fov,
                    "camera_lens_angle": camera_lens_angle
                },
                "projection_info": {
                    "is_perspective": viewport.IsPerspectiveProjection,
                    "camera_x": camera_x,
                    "camera_y": camera_y,
                    "camera_z": camera_z,
                    "camera_position": camera_location,
                    "view_transform": matrix_to_list(view_proj),
                    "world_to_camera": matrix_to_list(world_to_cam),
                    "camera_to_world": matrix_to_list(cam_to_world),
                    "view_to_screen": matrix_to_list(view_to_screen)
                }
            }
            
            camera_data["views"].append(view_data)
            
            # Capture RGB view
            view_capture = Rhino.Display.ViewCapture()
            view_capture.Width = image_width
            view_capture.Height = image_height
            view_capture.ScaleScreenItems = False
            view_capture.DrawAxes = False
            view_capture.DrawGrid = False
            view_capture.DrawGridAxes = False
            view_capture.TransparentBackground = False
            
            bitmap = view_capture.CaptureToBitmap(view)
            if bitmap:
                bitmap.Save(file_path, System.Drawing.Imaging.ImageFormat.Jpeg)
                print(f"Saved RGB: {file_path}")
                
                # Now capture the Z-buffer (depth information)
                try:
                    # Save the current display mode and settings
                    current_display_mode = viewport.DisplayMode
                    
                    # Use the ShowZBuffer command directly
                    # This command toggles the display of Z-buffer
                    rs.Command("_ShowZBuffer", False)
                    
                    # Give time for the command to execute and view to update
                    time.sleep(0.5)
                    
                    # Redraw the view to ensure Z-buffer visualization is applied
                    view.Redraw()
                    
                    # Create a depth capture view
                    depth_capture = Rhino.Display.ViewCapture()
                    depth_capture.Width = image_width
                    depth_capture.Height = image_height
                    depth_capture.ScaleScreenItems = False
                    depth_capture.DrawAxes = False
                    depth_capture.DrawGrid = False
                    depth_capture.DrawGridAxes = False
                    depth_capture.TransparentBackground = False
                    
                    # Capture the depth view
                    depth_bitmap = depth_capture.CaptureToBitmap(view)
                    
                    if depth_bitmap:
                        # Extract raw depth data from bitmap
                        raw_depth_data = extract_depth_data_from_bitmap(depth_bitmap)
                        
                        # Save normalized depth buffer to a numpy file alongside depth image
                        depth_npy_filename = f"view_elev{elevation}_angle{angle:.0f}_depth.npy"
                        depth_npy_path = os.path.join(output_folder, depth_npy_filename)
                        np.save(depth_npy_path, raw_depth_data)
                        
                        # Update view data with depth file information
                        view_data["depth_npy_filename"] = depth_npy_filename
                        
                        # Save the depth image as visible PNG
                        depth_bitmap.Save(depth_file_path, System.Drawing.Imaging.ImageFormat.Png)
                        print(f"Saved Depth: {depth_file_path}")
                        print(f"Saved Depth data: {depth_npy_path}")
                    else:
                        print(f"Failed to capture Z-buffer at elevation {elevation}, angle {angle}")
                    
                    # Turn off Z-buffer display by toggling the command again
                    rs.Command("_ShowZBuffer", False)
                    
                    # Restore the original display mode if needed
                    if viewport.DisplayMode != current_display_mode:
                        viewport.DisplayMode = current_display_mode
                        view.Redraw()
                
                except Exception as e:
                    print(f"Error capturing Z-buffer: {str(e)}")
                    # Attempt to turn off Z-buffer display in case of error
                    try:
                        rs.Command("_ShowZBuffer", False)
                    except:
                        pass
                
                image_count += 1
            else:
                print(f"Failed to capture view at elevation {elevation}, angle {angle}")
    
    # Save camera data to JSON file
    json_path = os.path.join(output_folder, "camera_data.json")
    with open(json_path, 'w') as json_file:
        json.dump(camera_data, json_file, indent=4)
    
    print(f"Capture complete. {image_count} RGB images and corresponding depth images saved to {output_folder}")
    print(f"Camera data saved to {json_path}")
    
    # Attempt to restore the original view
    try:
        rs.Command("_-View _Restore " + initial_view + " _Enter", False)
    except:
        print("Could not restore original view.")


def matrix_to_list(matrix):
    """Convert a Rhino Transform to a 4x4 list for JSON serialization"""
    return [
        [matrix.M00, matrix.M01, matrix.M02, matrix.M03],
        [matrix.M10, matrix.M11, matrix.M12, matrix.M13],
        [matrix.M20, matrix.M21, matrix.M22, matrix.M23],
        [matrix.M30, matrix.M31, matrix.M32, matrix.M33]
    ]


def extract_depth_data_from_bitmap(bitmap):
    """Extract depth values from Z-buffer visualization bitmap"""
    # Create array to hold pixel data
    rect = Rectangle(0, 0, bitmap.Width, bitmap.Height)
    bitmap_data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, 
                                  PixelFormat.Format24bppRgb)
    
    try:
        # Get pixel data
        byte_count = bitmap_data.Stride * bitmap_data.Height
        byte_array = System.Array.CreateInstance(System.Byte, byte_count)
        Marshal.Copy(bitmap_data.Scan0, byte_array, 0, byte_count)
        
        # Convert to numpy array
        pixel_values = np.zeros((bitmap.Height, bitmap.Width), dtype=np.float32)
        
        # Convert RGB values to depth values - Z-buffer visualizations typically use grayscale
        # where each pixel's RGB values are the same and represent the depth
        for y in range(bitmap.Height):
            for x in range(bitmap.Width):
                offset = y * bitmap_data.Stride + x * 3
                # In Z-buffer, typically all channels (R,G,B) have the same value
                # We'll use green channel as representative value
                green_value = byte_array[offset + 1] / 255.0  # Normalize to 0-1 range
                
                # In standard Z-buffer visualization, black (0) is near and white (1) is far
                # So we need to convert this to actual depth
                normalized_depth = green_value
                
                pixel_values[y, x] = normalized_depth
    
    finally:
        bitmap.UnlockBits(bitmap_data)
    
    return pixel_values


# Run the function
if __name__ == "__main__":
    capture_multi_view_screenshots()