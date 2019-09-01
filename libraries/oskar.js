// Version 1.0
// 26-08-2019

let canvas, ctx;

let width, height;

let windowWidth, windowHeight;

let mouseX, mouseY;
let keyCode;

let images = [];

const HALF_PI = Math.PI / 2;
const PI = Math.PI;
const TWO_PI = Math.PI * 2;
let frameCount = 0;

window.onload = function () {
    // canvas = document.createElement('canvas');
    // canvas.id = "canvas";
    // let gameArea = document.getElementById("gameArea");
    // gameArea.insertBefore(canvas, gameArea.firstChild);

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    // ctx.mozImageSmoothingEnabled = false;
    // ctx.webkitImageSmoothingEnabled = false;
    // ctx.msImageSmoothingEnabled = false;
    // ctx.imageSmoothingEnabled = false;
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.stroke();

    addEventListeners();

    setLibValues();

    setup(function () {
        setInterval(function () {
            draw();
            frameCount++;
        }, 1000 / 60);
    });
}

function redraw() {
    draw();
}

function setLibValues() {
    width = canvas.width;
    height = canvas.height;

    windowWidth = window.innerWidth;
    windowHeight = window.innerHeight;
}

let sources = {};

function imagesLoaded() {}

function loadImages(sources) {

    let numImages = 0;
    for (src in sources) {
        numImages++;
    }
    let loadedImages = 0;
    for (src in sources) {
        images[src] = new Image();
        images[src].onload = function () {
            loadedImages++;
            if (loadedImages == numImages) {
                //console.log("Images loaded!");
                imagesLoaded();
            }
        };
        images[src].src = sources[src];
    }
}

function image(img, x, y, width, height) {
    ctx.drawImage(img, x, y, width, height);
}

function resizeCanvas(width, height) {
    ctx.canvas.width = width;
    ctx.canvas.height = height;
    canvas.width = width;
    canvas.height = height;

    setLibValues();
}

function background(color) {
    fill(color);
    rect(0, 0, canvas.width, canvas.height);
}

function clearBackground() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function font(f) {
    ctx.font = f;
}

function textAlign(a) {
    ctx.textAlign = a;
}

function text(text, x, y) {
    ctx.fillText(text, x, y);
}

function randomRgb() {
    return rgba(random(0, 256), random(0, 256), random(0, 256));
}

function fill(color) {
    ctx.fillStyle = color;
}

function ellipse(x, y, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2, true);
    ctx.fill();
}

// function ellipse(x, y, rx, ry = rx) {
//     ctx.beginPath();
//     // ctx.arc(x, y, r, 0, Math.PI * 2, true);
//     ctx.ellipse(x, y, rx, ry, 0, 0, 2 * Math.PI);
//     ctx.stroke();
// }

function rect(x, y, width, height) {
    ctx.fillRect(x, y, width, height);
}

function strokeRect(x, y, width, height) {
    ctx.strokeRect(x, y, width, height);
}

function lineWidth(x) {
    ctx.lineWidth = x;
}

function line(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function globalAlpha(a = 1) {
    ctx.globalAlpha = a;
}

function rgb(r, g, b) {
    return rgba(r, g, b, 1);
}

function rgba(r, g, b, a = 1) {
    return "rgba(" + floor(r) + ", " + floor(g) + ", " + floor(b) + ", " + a + ")";
}

function hsl(h, s, l) {
    return hsla(h, s, l, 1);
}

function hsla(h, s, l, a = 1) {
    return "hsl(" + floor(h) + ", " + floor(s) + "%, " + floor(l) + "%, " + a + ")";
}

function lerp(start, end, rate) {
    return (end - start) * rate + start;
}

// function factorial(n) {
//     let res = 1;
//     for (let i = 2; i <=n; i++) {
//         res *= i;
//         // console.log(res);
//     }
//     return res;
// }

// MATH SHORTS

function pow(a, b) {
    return Math.pow(a, b);
}

function sqrt(a) {
    return Math.sqrt(a);
}

function floor(a) {
    return Math.floor(a);
}

function ceil(a) {
    return Math.ceil(a);
}

function round(a) {
    return Math.round(a);
}

function abs(a) {
    return Math.abs(a);
}

function random(a, b) {
    return Math.random() * (b - a) + a;
}

function min(a, b) {
    return a < b ? a : b;
}

function max(a, b) {
    return a > b ? a : b;
}

function map(x, fromA, fromB, toA, toB) {
    return (x - fromA) / (fromB - fromA) * (toB - toA) + toA;
}

// ARRAY PROTOTYPES

Array.prototype.random = function () {
    return this[Math.floor((Math.random() * this.length))];
}
Array.prototype.first = function () {
    return this[0];
}
Array.prototype.last = function () {
    return this[this.length - 1];
}

function push() {
    ctx.save()
};

function translate(x, y) {
    ctx.translate(x, y);
}

function pop() {
    ctx.restore();
}

function beginShape() {
    ctx.beginPath();
}

function vertex(x, y) {
    ctx.lineTo(x, y);
}

function endShape() {
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
}

function stroke(color) {
    ctx.strokeStyle = color;
}

function noStroke() {
    ctx.strokeStyle = "rgba(0, 0, 0, 0)";
}

function mouseContained(x1, y1, x2, y2) {
    return (mouseX > x1 && mouseX < x2 && mouseY > y1 && mouseY < y2);
}

// STORAGE PROTOTYPES

Storage.prototype.hasKey = function (key) {
    return this.getItem(key) !== null;
}
Storage.prototype.getParsed = function (key) {
    return JSON.parse(this.getItem(key));
}

// EVENT LISTENERS

function mousePressed() {}

function mouseReleased() {}

function windowResized() {}

function keyPressed() {}

function addEventListeners() {
    canvas.addEventListener("mousedown", mousePressed);
    canvas.addEventListener("mouseup", mouseReleased);
    canvas.addEventListener('mousemove', updateMouseMove);

    window.addEventListener("keypress", function (evt) {
        keyCode = evt.keyCode;
        keyPressed(keyCode);
    });

    window.addEventListener('touchstart', function (evt) {
        updateTouchMove(evt)
        mousePressed();
        evt.preventDefault();
    });
    window.addEventListener('touchend', mouseReleased);
    window.addEventListener('touchmove', function (evt) {
        evt.preventDefault();
        updateTouchMove(evt);
    });

    window.onresize = function (event) {
        setLibValues();
        windowResized();
    };

    function updateMouseMove(evt) {
        let rect = canvas.getBoundingClientRect();
        let root = document.documentElement;
        mouseX = evt.clientX - rect.left - root.scrollLeft;
        mouseY = evt.clientY - rect.top - root.scrollTop;
    }

    function updateTouchMove(evt) {
        let rect = canvas.getBoundingClientRect();
        let root = document.documentElement;
        mouseX = evt.touches[0].clientX - rect.left - root.scrollLeft;
        mouseY = evt.touches[0].clientY - rect.top - root.scrollTop;
    }
}

// VECTOR

class Vector {
    constructor(x = 0, y = 0) {
        this.x = x;
        this.y = y;
    }

    add(v) {
        this.x += v.x;
        this.y += v.y;
        return this;
    }

    sub(v) {
        this.x -= v.x;
        this.y -= v.y;
        return this;
    }

    mult(n) {
        this.x *= n;
        this.y *= n;
        return this;
    }

    div(n) {
        this.x /= n;
        this.y /= n;
        return this;
    }

    set(x, y) {
        this.x = x;
        this.y = y;
        return this;
    }

    copy() {
        return new Vector(this.x, this.y);
    }

    dot(v) {
        return this.x * v.x + this.y * v.y;
    }

    magSq() {
        return this.x * this.x + this.y * this.y;
    }

    mag() {
        return Math.sqrt(this.magSq());
    }

    normalize() {
        var len = this.mag();
        if (len !== 0) this.mult(1 / len);
        return this;
    }

    setMag(n) {
        this.normalize().mult(n);
        return this;
    }

    heading() {
        return Math.atan2(this.y, this.x);
    }

    rotate(a) {
        var newHeading = this.heading() + a;
        var mag = this.mag();
        this.x = Math.cos(newHeading) * mag;
        this.y = Math.sin(newHeading) * mag;
        return this;
    }
}

function vec(x = 0, y = 0) {
    return new Vector(x, y);
}

function distSq(x1, y1, x2, y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

function dist(x1, y1, x2, y2) {
    return Math.sqrt(distSq(x1, y1, x2, y2));
}

// MATRIX CLASS

class Matrix {
    constructor(rows = 2, cols = 2) {
        this.rows = rows;
        this.cols = cols;

        this.data = [];
        for (let i = 0; i < rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < cols; j++) {
                this.data[i][j] = 0;
            }
        }
    }

    each(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = func(this.data[i][j], i, j);
            }
        }
        return this;
    }

    copy() {
        let m = new Matrix(this.rows, this.cols);
        m.each((val, i, j) => this.data[i][j]);
        return m;
    }

    log() {
        console.table(this.data);
        return this;
    }

    fix() {
        this.rows = this.data.length;
        this.cols = this.data[0].length;
        return this;
    }

    randomize(a = 0, b = 1) {
        this.each(() => (Math.random() * (b - a) + a));
        return this;
    }

    add(n) {
        if (n instanceof Matrix) {
            if (this.rows != n.rows || this.cols != n.cols) {
                console.warn("Cols and rows of A must match cols and rows of B");
            } else {
                this.each((val, i, j) => val + n.data[i][j]);
            }
        } else {
            this.each((val) => val + n);
        }
        return this;
    }

    sub(n) {
        if (n instanceof Matrix) {
            if (this.rows != n.rows || this.cols != n.cols) {
                console.warn("Cols and rows of A must match cols and rows of B");
            } else {
                this.each((val, i, j) => val - n.data[i][j]);
            }
        } else {
            this.each((val) => val - n);
        }
        return this;
    }

    set(n) {
        if (n instanceof Matrix) {
            this.set(n.data);
        } else if (n instanceof Array) {
            this.data = n;
            this.fix();
        } else {
            this.each(() => n);
        }
        return this;
    }

    dot(n) {
        if (this.cols != n.rows) {
            console.warn("Cols of A must match rows of B");
        } else {
            // https://stackoverflow.com/questions/27205018/multiply-2-matrices-in-javascript
            let m = new Matrix(this.rows, n.cols);
            m.set(0);
            for (var i = 0; i < m.rows; i++) {
                for (var j = 0; j < m.cols; j++) {
                    for (var k = 0; k < this.cols; k++) {
                        m.data[i][j] += this.data[i][k] * n.data[k][j];
                    }
                }
            }
            this.data = m.data;
            this.fix();
        }
        return this;
    }

    mult(n) {
        if (n instanceof Matrix) {
            if (this.rows != n.rows || this.cols != n.cols) {
                console.warn("Cols and rows of A must match cols and rows of B");
            } else {
                // https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
                for (var i = 0; i < this.rows; i++) {
                    for (var j = 0; j < this.cols; j++) {
                        this.data[i][j] *= n.data[i][j];
                    }
                }
            }
        } else {
            this.each((val) => val * n);
        }
        return this;
    }

    transpose() {
        let m = new Matrix(this.cols, this.rows);
        m.each((val, i, j) => this.data[j][i]);
        this.set(m);
        return this;
    }

    static multiply(a, b) {
        if (a.cols != b.rows) {
            console.warn("Cols of A must match rows of B");
            return;
        }
        var m = new Matrix(a.rows, b.cols);
        m.set(0);
        for (var i = 0; i < a.rows; i++) {
            for (var j = 0; j < b.cols; j++) {
                for (var k = 0; k < a.cols; k++) {
                    m.data[i][j] += a.data[i][k] * b.data[k][j];
                }
            }
        }
        return m;
    }

    // todo refactor
    static fromArray(arr) {
        let m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    static toArray(m) {
        let res = [];
        for (let i = 0; i < m.data.length; i++) {
            for (let j = 0; j < m.data[0].length; j++) {
                res.push(m.data[i][j]);
            }
        }
        return res;
    }
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function d_sigmoid(y) {
    return y * (1 - y);
}

class NeuralNetwork {
    constructor(input_nodes, hidden_nodes, output_nodes) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        this.weights_ih = (new Matrix(this.hidden_nodes, this.input_nodes)).randomize(-1, 1);
        this.weights_ho = (new Matrix(this.output_nodes, this.hidden_nodes)).randomize(-1, 1);

        this.bias_h = (new Matrix(this.hidden_nodes, 1)).randomize(-1, 1);
        this.bias_o = (new Matrix(this.output_nodes, 1)).randomize(-1, 1);

        this.setLearningRate();

    }

    predict(input_array) {
        if (input_array.length != this.input_nodes) {
            console.error("WRONG INPUT_ARRAY LENGTH");
        }

        let inputs = Matrix.fromArray(input_array);
        // Generating the Hidden Outputs
        let hidden = Matrix.multiply(this.weights_ih, inputs).add(this.bias_h).each(sigmoid);
        // Generating the output's output!
        let outputs = Matrix.multiply(this.weights_ho, hidden).add(this.bias_o).each(sigmoid);

        return Matrix.toArray(outputs);
    }

    train(inputs_array, targets_array) {
        if (inputs_array.length != this.input_nodes) {
            console.error("WRONG INPUT_ARRAY LENGTH");
            return;
        } else if (targets_array.length != this.output_nodes) {
            console.error("WRONG TARGET_ARRAY LENGTH");
            return;
        }

        // Generating the Hidden Outputs
        let inputs = Matrix.fromArray(inputs_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs).add(this.bias_h).each(sigmoid);
        let outputs = Matrix.multiply(this.weights_ho, hidden).add(this.bias_o).each(sigmoid);

        // Convert array to matrix object
        let targets = Matrix.fromArray(targets_array);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        let output_errors = targets.copy().sub(outputs);

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        let gradients = outputs.copy().each(d_sigmoid).mult(output_errors).mult(this.learning_rate);

        let hidden_T = hidden.copy().transpose();
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_T);

        // adjust the weights weithsby deltas
        this.weights_ho.add(weights_ho_deltas);
        this.bias_o.add(gradients);

        // calc hidden layer errors
        let who_t = this.weights_ho.copy().transpose();
        let hidden_errors = Matrix.multiply(who_t, output_errors);

        // calculate hidden gradient
        let hidden_gradient = hidden.copy().each(d_sigmoid).mult(hidden_errors).mult(this.learning_rate);

        // calculate input->hidden deltas
        let inputs_T = inputs.copy().transpose();
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        // adjust the input->hidden weights
        this.weights_ih.add(weights_ih_deltas);
        this.bias_h.add(hidden_gradient);
    }

    setLearningRate(learning_rate = 0.1) {
        this.learning_rate = learning_rate;
    }
}