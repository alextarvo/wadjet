# ImageCutter

This program reads the image of the table with a single ball rolling on it. It detects the
ball using computer vision techniques (motion detection, blob detection), "cuts out" the image
of the ball along with its coordinates, and saves these into the file as protobufs. The protobuf
is defined in the cut_images.proto file.

# TrainingSetGenerator
This program loads a background video of the table and "cut out" images of the balls on that table,
# obtained previously by the ImageCutter.
# Then it starts inserting the randomly selected images of the balls at the background, to generate
# a realistic synthetic images of the table with balls on it. We insert the balls into original locations,
# because they also contain pieces of the table background. Inserting them into random locations will
# yield a "patchy" image of the table.
# The resulting data is saved into the Yolo dataset format.
