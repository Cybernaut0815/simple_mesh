# simple_mesh
SimpleMesh is a workflow that aims to create a clean model from a scanned/AI generated one. The model is rationalized and parsed using custom scripts and AI segmentation workflows. The resulting model is a NURBS Rhino model that can then be taken into Revit. 

The AI model generation was first done with Midjourney, to get an image, and then passed into Trelis and Polycam. Afterwards, the model was segmented using images or recorded videos. Using Rhino, we generate a multi-view that is passed through SAM 2 in order to get labels for the various elements: windows, doors, etc. The result is passed back into Rhino and projected back onto the mesh as a mask. The mask is used to parse the mesh which is then rationalized using Grasshopper.

Developed during the AECTech London Hackathon 2025 by Christoph Geiger, Thomas Lindemann, Renee Dobre, Jose Andres Amenabar and Alex Nap.
