"use strict";

var clickX = [];
var clickY = [];
var clickDrag = [];
var paint;

var fgcontext, fgcanvas, bgcanvas, bgcontext;

$(document).ready(function() {

    fgcanvas = $("canvas")[0];
    bgcanvas = $("#bgcanvas")[0];

    bgcontext = bgcanvas.getContext("2d");

    fgcontext = fgcanvas.getContext("2d");
    fgcontext.strokeStyle = "#ff0000";
    fgcontext.lineJoin = "round";
    fgcontext.lineWidth = 20;

    fgcanvas.addEventListener("mousedown", mouseWins);
    fgcanvas.addEventListener("touchstart", touchWins);

    $("#clear").click(onClickClear);
    $("#submit").click(onClickSubmit);
});

/**
 * Add information where the user clicked at.
 * @param {number} x
 * @param {number} y
 * @return {boolean} dragging
 */
function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

/**
 * Redraw the complete canvas.
 */
function redraw() {
    // Clears the canvas
    fgcontext.clearRect(0, 0, fgcontext.canvas.width, fgcontext.canvas.height);

    for (var i = 0; i < clickX.length; i += 1) {
        if (!clickDrag[i] && i == 0) {
            fgcontext.beginPath();
            fgcontext.moveTo(clickX[i], clickY[i]);
            fgcontext.stroke();
        } else if (!clickDrag[i] && i > 0) {
            fgcontext.closePath();

            fgcontext.beginPath();
            fgcontext.moveTo(clickX[i], clickY[i]);
            fgcontext.stroke();
        } else {
            fgcontext.lineTo(clickX[i], clickY[i]);
            fgcontext.stroke();
        }
    }
}

/**
 * Draw the newly added point.
 * @return {void}
 */
function drawNew() {
    var i = clickX.length - 1
    if (!clickDrag[i]) {
        if (clickX.length == 0) {
            fgcontext.beginPath();
            fgcontext.moveTo(clickX[i], clickY[i]);
            fgcontext.stroke();
        } else {
            fgcontext.closePath();

            fgcontext.beginPath();
            fgcontext.moveTo(clickX[i], clickY[i]);
            fgcontext.stroke();
        }
    } else {
        fgcontext.lineTo(clickX[i], clickY[i]);
        fgcontext.stroke();
    }
}

function mouseDownEventHandler(e) {
    paint = true;
    var x = e.pageX - fgcanvas.offsetLeft;
    var y = e.pageY - fgcanvas.offsetTop;
    if (paint) {
        addClick(x, y, false);
        drawNew();
    }
}

function touchstartEventHandler(e) {
    paint = true;
    if (paint) {
        addClick(e.touches[0].pageX - fgcanvas.offsetLeft, e.touches[0].pageY - fgcanvas.offsetTop, false);
        drawNew();
    }
}

function mouseUpEventHandler(e) {
    fgcontext.closePath();
    paint = false;
}

function mouseMoveEventHandler(e) {
    var x = e.pageX - fgcanvas.offsetLeft;
    var y = e.pageY - fgcanvas.offsetTop;
    if (paint) {
        addClick(x, y, true);
        drawNew();
    }
}

function touchMoveEventHandler(e) {
    if (paint) {
        addClick(e.touches[0].pageX - fgcanvas.offsetLeft, e.touches[0].pageY - fgcanvas.offsetTop, true);
        drawNew();
    }
}

function setUpHandler(isMouseandNotTouch, detectEvent) {
    removeRaceHandlers();
    if (isMouseandNotTouch) {
        fgcanvas.addEventListener("mouseup", mouseUpEventHandler);
        fgcanvas.addEventListener("mousemove", mouseMoveEventHandler);
        fgcanvas.addEventListener("mousedown", mouseDownEventHandler);
        mouseDownEventHandler(detectEvent);
    } else {
        fgcanvas.addEventListener("touchstart", touchstartEventHandler);
        fgcanvas.addEventListener("touchmove", touchMoveEventHandler);
        fgcanvas.addEventListener("touchend", mouseUpEventHandler);
        touchstartEventHandler(detectEvent);
    }
}

function mouseWins(e) {
    setUpHandler(true, e);
}

function touchWins(e) {
    setUpHandler(false, e);
}

function removeRaceHandlers() {
    fgcanvas.removeEventListener("mousedown", mouseWins);
    fgcanvas.removeEventListener("touchstart", touchWins);
}

function onClickClear() {
    fgcontext.clearRect(0, 0, fgcanvas.width, fgcanvas.height);
}

function onClickSubmit() {
    autocropCanvas();
    submitImage();
}

function autocropCanvas() {
    var width = fgcanvas.width,
        height = fgcanvas.height,
        minx = width,
        miny = height,
        maxx = 0,
        maxy = 0,
        imageData = fgcontext.getImageData(0, 0, width, height),
        x, y, index;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            index = (y * width + x) * 4;
            if (imageData.data[index + 3] > 0) {
                if (y < miny) miny = y;
                if (y > maxy) maxy = y;
                if (x < minx) minx = x;
                if (x > maxx) maxx = x;
            }
        }
    }

    // crop to writing
    var cropped = fgcontext.getImageData(minx, miny, maxx - minx, maxy - miny);
    bgcanvas.width = maxx - minx;
    bgcanvas.height = maxy - miny;

    // change non-opaque pixels to white
    var pixel_data = cropped.data;
    for (var i = 0; i < pixel_data.length; i += 4) {
        if (pixel_data[i + 3] < 255) {
            pixel_data[i] = 255 - pixel_data[i];
            pixel_data[i + 1] = 255 - pixel_data[i + 1];
            pixel_data[i + 2] = 255 - pixel_data[i + 2];
            pixel_data[i + 3] = 255 - pixel_data[i + 3];
        }
    }

    bgcontext.putImageData(cropped, 0, 0);
}

function submitImage() {
    var serialisedImage = bgcanvas
        .toDataURL()
        .substring(22)
        .replace(/\+/g, '-')
        .replace(/\//g, '_');

    $.ajax({
        url: "processimage.py",
        method: "POST",
        data: {
            "img": serialisedImage,
        },
        dataType: "json",
    }).done(function(data, textStatus) {
        console.log("Success:"); 
        console.log(data);
        console.log(textStatus);

        if (textStatus == "success") {
            $("#answer").html(data["prediction"]);
        }

    }).fail(function(data, textStatus, error) {
        console.log("Error:");
        console.log(data);
        console.log(textStatus);
        console.log(error);
    });
}
