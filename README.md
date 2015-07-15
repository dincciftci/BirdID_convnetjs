#BirdID_Convnetjs
----------

How to use:
----------
Format your dataset for use with process_images.py. It requires the images to be in the following folder structure:

    .
    |-- path_to_folders_with_images
    |    |-- class1
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    └ ...
    |    |-- class2
    |    |    └ ...
    |    |-- class3
        ...
    |    └-- classN
  
The code can be modified to create two versions of the dataset with different dimensions, to be used for augmentation.

- trainer.js: Define the network and train it

- tester.js: Run a trained network and output accuracy. Prints per-category accuracy and the accuracy of the top 2 and top 3 guesses as well.


Dependencies:
----------

- [Node.js][4]
- [convnetjs][1]
- [graceful-fs][2]
- [node-canvas][3]

[1]: https://github.com/karpathy/convnetjs
[2]: https://github.com/isaacs/node-graceful-fs
[3]: https://github.com/Automattic/node-canvas
[4]: https://nodejs.org/
