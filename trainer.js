var convnetjs = require("convnetjs");
var fs = require("graceful-fs");
var Canvas = require('canvas');
var zlib = require('zlib');
var ts = require('./utilities.js');//tools are being referenced from the utilities.js file within the same directory

var birdpath = "../cnn_dataset/160_auto/";
var valpath = "../cnn_dataset/128_auto/";//folder to validate from
//two folders are needed because the training set is larger but augmented


var ratios = [0.6, 0.4]//training - validation - testing ratios
//testing images can be omitted

var debug = false;//enable to output which image is being trained with and the accuracy of the prediction on that specific image

var checkpoint = 5;//save network state and calculate validation accuracy every x runs

var preaug_dim = 160;//dimension of the the training images before augmentation
var dim = 128;//dimension of the images after augmentation (default size of the validation images)
var per_category = 98;//images per category in both folders
var categories = 9;//number of categories


//convnetjs' utilities for visualization
var cnnutil = require('./cnnutil.js');
var cnnvis = require('./cnnvis.js');


var canvas = new Canvas(dim, dim); //all images are 512x512
var ctx = canvas.getContext('2d');
var Image = Canvas.Image;

global.canvas = canvas;
global.ctx = ctx;
global.Image = Image;

var ID = Date.now();//will be used in file names to uniquely identify this run

var drawImg = ts.drawImg;
var img_to_vol = ts.img_to_vol;
var split_dataset = ts.split_dataset;
var getSVMPrediction = ts.getSVMPrediction;

//CNN Definition starts here

var layer_defs = [];

layer_defs.push({type:'input', out_sx:dim, out_sy:dim, out_depth:3}); // declare size of input

layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});

layer_defs.push({type:'conv', sx:5, filters:32, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});


layer_defs.push({type:'conv', sx:5, filters:64, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});

layer_defs.push({type:'softmax', num_classes:categories});

net = new convnetjs.Net();
net.makeLayers(layer_defs);

trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:1, l2_decay:0.0001});



//Generate array of file names and labels
var all_names = [ ];
var all_labels = [ ];
for (var i = 0; i < per_category * categories; i++)
{
  all_names.push("bird" + i + ".png");
  all_labels.push(Math.floor(i / per_category));
}

var split = split_dataset(all_names, all_labels, ratios);
var names = split[0];
var labels = split[1];

//save the split for future reference for testing
var json = {};
json.data = split;
fs.writeFileSync("splits/cnn_split" + ID + ".txt", JSON.stringify(json) );


//Create windows to log accuracy
var valAccWindow = new cnnutil.Window(names[1].length);
var testAccWindow = new cnnutil.Window(names[0].length, 1);


console.log("[" + ID + "] Starting training...");
console.time("Training time");

//epochs
for (var e = 0; e < 301; e++)
{

  //convert each image to vols and train with them in batches
  for (var i = 0; i < names[0].length; i++)//loops batches
  {

    if (Math.floor(Math.random() * 2) == 1)//50% chance of augmentation
    {
      var canvas = new Canvas(preaug_dim, preaug_dim); 
      var ctx = canvas.getContext('2d');
      global.canvas = canvas;
      global.ctx = ctx;

      var x = img_to_vol(birdpath + names[0][i]);


      //augment
      //random crops and flips
      var canvas = new Canvas(dim, dim); 
      var ctx = canvas.getContext('2d');
      global.canvas = canvas;
      global.ctx = ctx;

      x = convnetjs.augment(x, dim, Math.floor(Math.random() * (preaug_dim - dim)), Math.floor(Math.random() * (preaug_dim - dim)), Math.floor(Math.random() * 2) == 1);

    } else {

      var canvas = new Canvas(dim, dim); 
      var ctx = canvas.getContext('2d');
      global.canvas = canvas;
      global.ctx = ctx;
      var x = img_to_vol(valpath + names[0][i]);
      //50% chance of flipping
      
      if (Math.floor(Math.random() * 2) == 1)
        x = convnetjs.augment(x, dim, 0, 0, true);

    }

    var output = trainer.train(x, labels[0][i]);

    fs.appendFileSync("logs/cnn_log" + ID + ".txt", "Epoch # "+ (e + 1)  + ", Run #" + (i + 1) + "\n" + JSON.stringify(output) + "\n");

    //resetting fs to avoid "too many open files" errors
    fs = [];
    fs = require("graceful-fs");

    //if the classifier layer is an SVM, this can be used
    //var pred = getSVMPrediction(net.layers[net.layers.length - 1]);
    var pred = net.getPrediction();
    var acc = pred==labels[0][i] ? 1.0 : 0.0;
    testAccWindow.add(acc);


    if(debug){

      var chance = net.layers[net.layers.length - 1].out_act.w[labels[0][i]];
      console.log("Trained image number " +i+ " with label " + labels[0][i] + ", chance: " + chance);
    }

  }


  if (Math.floor(e / 5) == (e/5))
    console.log("Finished epoch #" + (e+1));


  console.log("Training accuracy: " +testAccWindow.get_average());
  testAccWindow.reset();



  //Save network state every x epochs
  if (Math.floor(e/checkpoint) == (e/checkpoint) && e != 0)
  {

    var json = net.toJSON();
    var str = JSON.stringify(json);
    //var gzip = zlib.createGzip({ level: 9 });


    var filename = "states/cnn_state_" + ID +"_tmp.txt";
    fs.writeFileSync(filename, str);
    
    //network states can be compressed using the code below, but thisis done asynchoronously and is therefore really only useful for a final save
    
    //var rStream = fs.createReadStream(filename);
    //var wStream = fs.createWriteStream(filename + ".gz");

    /*rStream.pipe(gzip).pipe(wStream).on('finish', function () {
      fs.unlinkSync(filename);//remove the huge plaintext file
      });
      */


    //validate here

    for (var i = 0; i < names[1].length; i++)//loops batches
    {
      var x = img_to_vol(valpath + names[1][i]);
      var output = net.forward(x);

      //pred = getSVMPrediction(net.layers[net.layers.length - 1]);

      var pred = net.getPrediction();
      acc = pred==Math.floor(labels[1][i]) ? 1.0 : 0.0;
      valAccWindow.add(acc);
    }

    console.log("Validation accuracy: " + valAccWindow.get_average());

    valAccWindow.reset();

  }




}

console.timeEnd("Training time");

//Save the network
var json = net.toJSON();
var str = JSON.stringify(json);

//Date.now() to prevent overwriting states
filename = "states/cnn_state_" + ID +".txt";
fs.writeFileSync(filename, str);

//compression code below if needed
/*
   rStream = fs.createReadStream(filename);
   wStream = fs.createWriteStream(filename + ".gz");

   rStream.pipe(gzip)//compress
   .pipe(wStream)//write
   .on('finish', function () {  // finished
   fs.unlinkSync(filename);//remove the huge plaintext file
   }); */
