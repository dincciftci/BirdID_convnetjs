var convnetjs = require("convnetjs");
var fs = require("graceful-fs");
var Canvas = require('canvas');
var zlib = require('zlib');
var ts = require('./utilities.js');

//path containing the validation images
var birdpath = "../cnn_dataset/128/";

var retrainID = 1435340786541;//ID of the instance being retrained

var test_type = 2;//0 for training set, 1 for validation set, 2 for testing test
//there might not be a separate "testing" set because only training and validation splits are needed to train

var debug = false;


var dim = 128;//dimension of the images

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

var img_to_vol = ts.img_to_vol;

//CNN Definition
//load from state
var state = fs.readFileSync("states/cnn_state_" + retrainID + "_tmp.txt");
var net = new convnetjs.Net();
net.fromJSON(JSON.parse(state));

//Generate array of file names and labels
var split = JSON.parse(fs.readFileSync("splits/cnn_split" + retrainID + ".txt", "utf8" )).data;
var names = split[0];
var labels = split[1];


var valAccWindow = new cnnutil.Window(names[test_type].length);
var top2AccWindow = new cnnutil.Window(names[test_type].length);
var top3AccWindow = new cnnutil.Window(names[test_type].length);

//Tracking accuracy by category
var bird_accuracies = { };

for (var i = 0; i < net.layers[net.layers.length - 1].out_depth; i++)
{

  bird_accuracies[i] = new cnnutil.Window(names[test_type].length, 1);//this assigns too many spaces on the window but it will not cause any problems

}

console.log("[" + retrainID + "] Starting testing...");
console.time("Testing time");

for (var i = 0; i < names[test_type].length; i++)//loops batches
  {
    var x = img_to_vol(birdpath + names[test_type][i]);
    var output = net.forward(x);

    pred = net.getPrediction();
    acc = pred == labels[test_type][i] ? 1.0 : 0.0;
    valAccWindow.add(acc);
    bird_accuracies[labels[test_type][i]].add(acc);

    pred2 = ts.getTopPrediction(net.layers[net.layers.length - 1], 2);
    if(pred2[0] == labels[test_type][i] || pred2[1] == labels[test_type][i])
      top2AccWindow.add(1.0);
    else
      top2AccWindow.add(0.0);

    pred3 = ts.getTopPrediction(net.layers[net.layers.length - 1], 3);
    if(pred3[0] == labels[test_type][i] || pred3[1] == labels[test_type][i] || pred3[2] == labels[test_type][i])
      top3AccWindow.add(1.0);
    else
      top3AccWindow.add(0.0);

  }

console.log("Testing accuracy: " + valAccWindow.get_average());
console.log("Top 2 accuracy:" + top2AccWindow.get_average());
console.log("Top 3 accuracy:" + top3AccWindow.get_average());


for (var i = 0; i < net.layers[net.layers.length - 1].out_depth; i++)
{

  console.log("Accuracy for category " + i + " [" + bird_accuracies[i].v.length + "]: " + bird_accuracies[i].get_average());

}

console.timeEnd("Testing time");
