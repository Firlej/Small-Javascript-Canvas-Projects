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