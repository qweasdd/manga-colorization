var canvas = document.getElementById('draw_canvas');
var ctx = canvas.getContext('2d');
var canvasWidth = canvas.width;
var canvasHeight = canvas.height;
var prevX, prevY;

var result_canvas = document.getElementById('result');
var result_ctx = result_canvas.getContext('2d');
result_canvas.width = canvas.width;
result_canvas.height = canvas.height;

var color_indicator = document.getElementById('color');
ctx.fillStyle = 'black';
color_indicator.value = '#000000';

var cur_id = window.location.pathname.substring(window.location.pathname.lastIndexOf('/') + 1);

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

var init_hint = new Image();  
init_hint.addEventListener('load', function() {
      ctx.drawImage(init_hint, 0, 0);
});
init_hint.src =  '../static/temp_images/' + cur_id + '/hint.png?' + getRandomInt(100000).toString();

result_canvas.addEventListener('load', function(e) {
    var img = new Image();   
    img.addEventListener('load', function() {
      ctx.drawImage(img, 0, 0);
    }, false);
    console.log(window.location.pathname);
})


canvas.onload = function (e) {
    var img = new Image();   
    img.addEventListener('load', function() {
      ctx.drawImage(img, 0, 0);
    }, false);
    console.log(window.location.pathname);
   //img.src = ;
}

function reset() {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: (evt.clientX - rect.left) / (rect.right - rect.left) * canvas.width,
        y: (evt.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height
    };
}

function colorize() {
    var file_id = document.location.pathname;
    var image = canvas.toDataURL();
    
    $.post("/colorize", { save_file_id: file_id, save_image: image}).done(function( data ) {
        //console.log(document.location.origin + '/img/' + data)
       //window.open(document.location.origin + '/img/' + data, '_blank');
        //result.src = data;
        var img = new Image();   
        img.addEventListener('load', function() {
          result_ctx.drawImage(img, 0, 0);
        }, false);
        img.src = data;
   });
}

canvas.addEventListener('mousedown', function(e) {
    var mousePos = getMousePos(canvas, e);
    if (e.button == 0) {
        ctx.fillRect(mousePos['x'], mousePos['y'], 1, 1);
    }
    
    if (e.button == 2) {
       prevX = mousePos['x']
       prevY = mousePos['y']
    }
    
})

canvas.addEventListener('mouseup', function(e) {
    if (e.button == 2) {
        var mousePos = getMousePos(canvas, e);
        var diff_width = mousePos['x'] - prevX;
        var diff_height = mousePos['y'] - prevY;
        
        ctx.clearRect(prevX, prevY, diff_width, diff_height);
    }
})


canvas.addEventListener('contextmenu', function(evt) {
    evt.preventDefault();
})

function color(color_value){
    ctx.fillStyle = color_value;
    color_indicator.value = color_value;
}   

color_indicator.oninput = function() {
    color(this.value);
}

function rgbToHex(rgb){
  return '#' + ((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]).toString(16);
};

result_canvas.addEventListener('click', function(e) {
    if (e.button == 0) { 
        var cur_pixel = result_ctx.getImageData(e.offsetX, e.offsetY, 1, 1).data;
        color(rgbToHex(cur_pixel));
    }
    })
