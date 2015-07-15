var utilities = (function(exports){

  var convnetjs = require("convnetjs");
  var fs = require("fs");
  var Canvas = require('canvas');

  //used as the callback in the img_to_vol method to draw the given picture on the canvas
  var drawImg = function(err, pic){
    if (err) throw err;
    img = new Image;
    img.src = pic;
    ctx.drawImage(img, 0, 0);
  };


  //adapted from convnetjs.img_to_vol
  //converts the image from the given path into a Vol object to be used by the CNN
  var img_to_vol = function(img_path) {

    //fs.readFile(img_path, drawImg);

    img = new Image;
    img.src = fs.readFileSync(img_path);
    ctx.drawImage(img, 0, 0);

    try {
      var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    } catch (e) {

      if(e.name === 'IndexSizeError') {
        return false;
      } else {
        throw e;
      }
    }

    // prepare the input: get pixels and normalize them
    var p = img_data.data;
    var W = img.width;
    var H = img.height;
    var pv = []

      for(var i=0;i<p.length;i++) {
        pv.push(p[i]/255.0-0.5); // normalize image pixels to [-0.5, 0.5]
      }

    var x = new convnetjs.Vol(W, H, 4, 0.0); //input volume (image)
    x.w = pv;

    return x;

  }


  //Draws the activations of A (of type Vol)
  //returns an RGB canvas
  var draw_activations = function(A, scale) {

    var s = scale || 2; // scale
    // get max and min activation to scale the maps automatically
    var w = A.w;
    var mm = convnetjs.maxmin(w);

    var result = [ ];

    var canv = new Canvas(32, 32); //all images are 512x512
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);

    for(var d=0;d<3;d++) {
      for(var x=0;x<A.sx;x++) {
        for(var y=0;y<A.sy;y++) {
          var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  

          for(var dx=0;dx<s;dx++) {
            for(var dy=0;dy<s;dy++) {
              var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
              g.data[pp + d] = dval;
              if(d===0) g.data[pp+3] = 255; // alpha channel
            }
          }
        }
      }
      //result.push(canv);
    }
    ctx.putImageData(g, 0, 0);
    return canv;
  }

  //splits the given arrays of names and labels into groups based on the provided ratios
  //returns a 3D array with the structure [[names], [labels]]
  // the 2d arrays have their contents split into groups according to the ratios
  var split_dataset = function(names, labels, ratios) {

    if (!(names instanceof Array && labels instanceof Array && ratios instanceof Array)){
      return;
    }

    //check if ratios sum to 1
    var sum = 0;
    for (var i = 0; i < ratios.length; i++) {

      sum = sum + ratios[i];
    }

    if (sum !== 1)
      return;


    var total = names.length;
    var amounts = [ ];//to store the amount of elemets to have in each group


    for (var i = 0; i < ratios.length; i++) {

      amounts.push(Math.floor(total * ratios[i]));

    }

    //make sure the assigned numbers add up to the sum

    sum = 0;
    for (var i = 0; i < amounts.length; i++) {

      sum = sum + amounts[i];
    }
    //if too few items assigned, increment the last one
    if (sum < total){
      while (sum < total){
        amounts[amounts.length - 1]++;
        sum++;
      }
    } else if (sum > total) {//if too many, decrement the last one
      while (sum > total) {
        amounts[amounts.length - 1]--;
        sum--;
      }
    }

    //randomly assign names to each group
    var name_clone = names.slice(0);
    var label_clone = labels.slice(0);
    var result = [[ ], [ ] ];
    //names, labels

    for (var i = 0; i < amounts.length; i++){
      result[0].push([ ]);//initializing the corresponding arrays
      result[1].push([ ]);
      for (var j = 0; j < amounts[i]; j++){
        var index = Math.floor(Math.random() * name_clone.length);
        result[0][i].push(name_clone.splice(index, 1)[0]);
        result[1][i].push(label_clone.splice(index, 1)[0]);//remove one element at the calculated random index
        //the splice method returns an array of one item, so that item is pushed to the appropriate index in the result array
      }
    }

    return result;
  }

  //given the softmax layer, returns an array of the top predictions from the layer, in desencding order. "amount" determines how many predictions to return.
  var getTopPrediction = function(softmax_layer, amount) {

    //assert(softmax_layer.layer_type === 'softmax', 'getTopPrediction() requires a softmax layer');
    
    if (softmax_layer.layer_type !== 'softmax')
      return;

    var p = Object.keys(softmax_layer.out_act.w).map(function(k) { return softmax_layer.out_act.w[k] });//made into an array

    var result = [ ];

    if (p.length < amount || amount === 0)//requested too many
      return;

    for (var a = 0; a < amount; a++) {

      var maxV = p[0];
      var maxi = 0;

      for (var i = 1; i < p.length; i++) {

        if (p[i] > maxV) {
	  maxV = p[i];
	  maxi = i;
	}
      }

      result.push(maxi);
      p[maxi] = 0;//set the element to zero so that it is not picked up again

    }
    return result;
  }

  var getSVMPrediction = function (svm_layer) {

    if (svm_layer.layer_type !== 'svm')
      return;

    var p = svm_layer.out_act.w;
    var maxV = p[0];
    var maxi = 0;

    for (var i = 1; i < p.length; i++) {

      if (p[i] > maxV) {
        maxV = p[i];
        maxi = i;
      }
    }

    return maxi;//returns the index of the class with the highest score
  }

  exports = exports || {};
  exports.img_to_vol = img_to_vol;
  exports.drawImg= drawImg;
  exports.draw_activations = draw_activations;
  exports.split_dataset = split_dataset;
  exports.getTopPrediction = getTopPrediction;
  exports.getSVMPrediction = getSVMPrediction;
  return exports;

})(module.exports);//added to exports
