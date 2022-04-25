# Actin Comet Tail Quantification

For the quantitative analysis of the actin comet tail images, we developed an image annotation tool and a custom analysis workflow to extract mophorlogical information from the images.

The image annotation tool is built on top of ImJoy (https://imjoy.io, a web-based computational platform for developing interactive data analysis tools) and Kaibu (https://kaibu.org, a browser-based image annotation plugin for ImJoy). The multi-channel actin comet tail images are preprocessed to render as color images in PNG format and hosted on a server, then a sharable link was generated for remote annotation of the comet tails. During the annotation, the annotators were instructed to draw polygons around the tail for both the actin channel and the Arp3 channel. The annotation are stored in GeoJSON format on the server and processed after the annotation.

A python script was created for performing the image quantification in order to extract the mean curvature, length and area of each annotated actin comet tail. It is done by generating a label image of all the comet tail in the microscopy image. Each comet tail in the image were assigned with a unique ID (from 1 to the total number of comet tails in the image) and the result label image was generated by filling each comet tail polygon with a pixel value equals to its ID. The label image with the tails are then processed with mainly two python modules: scikit-image (https://scikit-image.org/, a Python image analysis library) and skan(https://skeleton-analysis.org/, a Python library for analysing skeleton images). More specifically, we used skimage.measure.regionprops to compute the pixel area of each act comet tail. For computing the length and curvature of the actin comet tail, the label images were skeletonize with the skimage.morphology.skeletonize function. For exach skeletonized tail, a line path was extracted with the member functions under each skan.Skeleton object. The path was then used to compute the mean curvature and length for each tail.


## Data Availability

The actin comet tail images along with the annotations will be publicly accessible via Zenodo.

The source code for performing the actin comet tail quantification are freely accessbile via Github: https://github.dev/radoshevichlab/YZhang