/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Gradient;
using numpy;
using tensorflow;
using tensorflow.python.ops.variables;
using tensorflow.train;
using static System.Linq.Enumerable;

namespace TensorFlowNET.Examples
{
    using NumSharp;
    using Python.Runtime;
    using Tensorflow.Hub;
    using np = numpy.np;

    /// <summary>
    /// Convolutional Neural Network classifier for Hand Written Digits
    /// CNN architecture with two convolutional layers, followed by two fully-connected layers at the end.
    /// Use Stochastic Gradient Descent (SGD) optimizer. 
    /// https://www.easy-tensorflow.com/tf-tutorials/convolutional-neural-nets-cnns/cnn1
    /// </summary>
    public class DigitRecognitionCNN
    {
        const string Name = nameof(DigitRecognitionCNN);
        string logs_path = "logs";

        const int img_h = 28, img_w = 28; // MNIST images are 28x28
        int n_classes = 10; // Number of classes, one class per digit
        int n_channels = 1;

        // Hyper-parameters
        int epochs = 5; // accuracy > 98%
        int batch_size = 100;
        float learning_rate = 0.001f;
        Datasets<MnistDataSet> mnist;

        // Network configuration
        // 1st Convolutional Layer
        int filter_size1 = 5;  // Convolution filters are 5 x 5 pixels.
        int num_filters1 = 16; //  There are 16 of these filters.
        int stride1 = 1;  // The stride of the sliding window

        // 2nd Convolutional Layer
        int filter_size2 = 5; // Convolution filters are 5 x 5 pixels.
        int num_filters2 = 32;// There are 32 of these filters.
        int stride2 = 1;  // The stride of the sliding window

        // Fully-connected layer.
        int h1 = 128; // Number of neurons in fully-connected layer.

        Tensor x, y;
        Tensor loss, accuracy, cls_prediction;
        Operation optimizer;

        int display_freq = 100;
        float accuracy_test = 0f;
        float loss_test = 1f;

        ndarray<float> x_train;
        ndarray<float> y_train;
        ndarray x_valid, y_valid;
        ndarray x_test, y_test;
        static PyObject numPy;

        public DigitRecognitionCNN()
        {
            using var _ = Py.GIL();
            if (numPy is null)
                numPy = PythonEngine.ImportModule("numpy");
        }

        public bool Run()
        {
            PrepareData();

            Train();
            // Test();
            
            return accuracy_test > 0.98;
        }

        public Graph BuildGraph()
        {
            var graph = new Graph();
            graph.as_default_dyn().__enter__();

            using (new name_scope("Input").StartUsing())
            {
                // Placeholders for inputs (x) and outputs(y)
                x = tf.placeholder(tf.float32, new TensorShape(null, img_h, img_w, n_channels), name: "X");
                y = tf.placeholder(tf.float32, new TensorShape(null, n_classes), name: "Y");
            }

            var conv1 = conv_layer(x, filter_size1, num_filters1, stride1, name: "conv1");
            var pool1 = max_pool(conv1, ksize: 2, stride: 2, name: "pool1");
            var conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name: "conv2");
            var pool2 = max_pool(conv2, ksize: 2, stride: 2, name: "pool2");
            var layer_flat = flatten_layer(pool2);
            var fc1 = fc_layer(layer_flat, h1, "FC1", use_relu: true);
            var output_logits = fc_layer(fc1, n_classes, "OUT", use_relu: false);

            using (new variable_scope("Train").StartUsing())
            {
                using (new variable_scope("Loss").StartUsing())
                {
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: output_logits), name: "loss");
                }

                using (new variable_scope("Optimizer").StartUsing())
                {
                    optimizer = new AdamOptimizer(learning_rate: learning_rate, name: "Adam-op").minimize(loss);
                }

                using (new variable_scope("Accuracy").StartUsing())
                {
                    var correct_prediction = tf.equal_dyn(tf.argmax(output_logits, 1), tf.argmax(y, 1), name: "correct_pred");
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name: "accuracy");
                }

                using (new variable_scope("Prediction").StartUsing())
                {
                    cls_prediction = tf.argmax(output_logits, axis: 1, name: "predictions");
                }
            }

            return graph;
        }

        public void Train()
        {
            Graph graph = BuildGraph();
            var sess = new Session(graph: graph);
            using (sess.StartUsing())
            {
                // Number of training iterations in each epoch
                int num_tr_iter = y_train.shape.Item1 / batch_size;

                var init = tf.global_variables_initializer();
                sess.run(init);

                float loss_val = 100.0f;
                float accuracy_val = 0f;

                var sw = new Stopwatch();
                sw.Start();
                foreach (var epoch in Range(0, epochs))
                {
                    Console.WriteLine($"Training epoch: {epoch + 1}");
                    Shuffle(x_train, y_train);

                    foreach (var iteration in Range(0, num_tr_iter))
                    {
                        var start = iteration * batch_size;
                        var end = (iteration + 1) * batch_size;
                        var (x_batch, y_batch) = GetNextBatch(x_train, y_train, start, end);

                        // Run optimization op (backprop)
                        sess.run(optimizer, feed_dict: new Dictionary<object,object> {
                            [x] = x_batch,
                            [y] = y_batch,
                        });

                        if (iteration % display_freq == 0)
                        {
                            // Calculate and display the batch loss and accuracy
                            IList<float32> stats = sess.run(new []{ loss, accuracy }, new Dictionary<object,object>{
                                [x] = x_batch,
                                [y] = y_batch,
                            });
                            loss_val = stats[0];
                            accuracy_val = stats[1];
                            Console.WriteLine($"iter {iteration.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")} {sw.ElapsedMilliseconds}ms");
                            sw.Restart();
                        }
                    }

                    // Run validation after every epoch
                    IList<float32> validationStats = sess.run(new []{loss, accuracy}, feed_dict: new Dictionary<object,object> {
                        [x] = this.x_valid,
                        [y] = this.y_valid,
                    });
                    loss_val = validationStats[0];
                    accuracy_val = validationStats[1];
                    Console.WriteLine("---------------------------------------------------------");
                    Console.WriteLine($"Epoch: {epoch + 1}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}");
                    Console.WriteLine("---------------------------------------------------------");
                }

                SaveCheckpoint(sess);
            }
        }

        public void Test()
        {
            var graph = new Graph();
            graph.as_default_dyn().__enter__();
            var sess = new Session(graph: graph);
            using (sess.StartUsing())
            {
                var saver = tf.train.import_meta_graph(Path.Combine(Name, "mnist_cnn.ckpt.meta"));
                // Restore variables from checkpoint
                saver.restore(sess, tf.train.latest_checkpoint(Name));

                loss = graph.get_tensor_by_name_dyn("Train/Loss/loss:0");
                accuracy = graph.get_tensor_by_name_dyn("Train/Accuracy/accuracy:0");
                x = graph.get_tensor_by_name_dyn("Input/X:0");
                y = graph.get_tensor_by_name_dyn("Input/Y:0");

                //var init = tf.global_variables_initializer();
                //sess.run(init);

                IList<float32> stats = sess.run(new []{loss, accuracy}, feed_dict: new Dictionary<object,object> {
                    [x] = this.x_test,
                    [y] = this.y_test,
                });
                loss_test = stats[0];
                accuracy_test = stats[1];
                Console.WriteLine("---------------------------------------------------------");
                Console.WriteLine($"Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}");
                Console.WriteLine("---------------------------------------------------------");
            }
        }

        /// <summary>
        /// Create a 2D convolution layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="filter_size">size of each filter</param>
        /// <param name="num_filters">number of filters(or output feature maps)</param>
        /// <param name="stride">filter stride</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor conv_layer(Tensor x, int filter_size, int num_filters, int stride, string name)
        {
            using(new variable_scope(name).StartUsing())
            {
                TensorShape inputShape = x.shape;
                Dimension num_in_channel = inputShape.dims[x.shape.ndims - 1];
                var shape = new TensorShape(filter_size, filter_size, num_in_channel.value, num_filters);
                var W = weight_variable("W", shape);
                // var tf.summary.histogram("weight", W);
                var b = bias_variable("b", new[] { num_filters });
                // tf.summary.histogram("bias", b);
                Tensor layer = tf.nn.conv2d_dyn(x, W,
                                     strides: new[] { 1, stride, stride, 1 },
                                     padding: "SAME");
                layer += b;
                return tf.nn.relu(layer);
            }
        }

        /// <summary>
        /// Create a max pooling layer
        /// </summary>
        /// <param name="x">input to max-pooling layer</param>
        /// <param name="ksize">size of the max-pooling filter</param>
        /// <param name="stride">stride of the max-pooling filter</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor max_pool(Tensor x, int ksize, int stride, string name)
        {
            return tf.nn.max_pool_dyn(x,
                ksize: new[] { 1, ksize, ksize, 1 },
                strides: new[] { 1, stride, stride, 1 },
                padding: "SAME",
                name: name);
        }

        /// <summary>
        /// Flattens the output of the convolutional layer to be fed into fully-connected layer
        /// </summary>
        /// <param name="layer">input array</param>
        /// <returns>flattened array</returns>
        private Tensor flatten_layer(Tensor layer)
        {
            using (new variable_scope("Flatten_layer").StartUsing())
            {
                TensorShape layer_shape = layer.shape;
                var num_features = layer_shape.dims.Skip(1).Aggregate(1, (s, dim) => s * (int)dim.value);
                var layer_flat = tf.reshape(layer, shape: new []{-1, num_features});

                return layer_flat;
            }
        }

        /// <summary>
        /// Create a weight variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private RefVariable weight_variable(string name, TensorShape shape)
        {
            var initer = new truncated_normal_initializer(stddev: 0.01f);
            return tf.get_variable(name,
                                   dtype: tf.float32,
                                   shape: shape,
                                   initializer: initer);
        }

        /// <summary>
        /// Create a bias variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private RefVariable bias_variable(string name, int[] shape)
        {
            var initial = tf.constant(0f, shape: shape, dtype: tf.float32);
            return tf.get_variable(name,
                           dtype: tf.float32,
                           initializer: initial);
        }

        /// <summary>
        /// Create a fully-connected layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="num_units">number of hidden units in the fully-connected layer</param>
        /// <param name="name">layer name</param>
        /// <param name="use_relu">boolean to add ReLU non-linearity (or not)</param>
        /// <returns>The output array</returns>
        private Tensor fc_layer(Tensor x, int num_units, string name, bool use_relu = true)
        {
            using (new variable_scope(name).StartUsing())
            {
                int? in_dim = x.shape[1].value;

                var W = weight_variable("W_" + name, new TensorShape(in_dim, num_units));
                var b = bias_variable("b_" + name, new[] { num_units });

                var layer = tf.matmul(x, W) + b;
                if (use_relu)
                    layer = tf.nn.relu(layer);

                return layer;
            }
        } 
            
        public void PrepareData()
        {
            Directory.CreateDirectory(Name);

            mnist = MnistModelLoader.LoadAsync(".resources/mnist", oneHot: true, showProgressInConsole: true).Result;
            (x_train, y_train) = Reformat(mnist.Train.Data, mnist.Train.Labels);
            (x_valid, y_valid) = Reformat(mnist.Validation.Data, mnist.Validation.Labels);
            (x_test, y_test) = Reformat(mnist.Test.Data, mnist.Test.Labels);

            Console.WriteLine("Size of:");
            Console.WriteLine($"- Training-set:\t\t{mnist.Train.Data.Shape.Size}");
            Console.WriteLine($"- Validation-set:\t{mnist.Validation.Data.Shape.Size}");
        }

        /// <summary>
        /// Reformats the data to the format acceptable for convolutional layers
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        private (ndarray<float>, ndarray<float>) Reformat(NDArray x, NDArray y)
        {
            var (img_size, num_ch, num_class) = ((int)Math.Sqrt(x.shape[1]), 1, NumSharp.np.unique(NumSharp.np.argmax(y, 1)).Shape.Size);
            var dataset = (ndarray<float>)ToGradient<float>(x).reshape(new int[]{x.shape[0], img_size, img_size, num_ch}).astype(np.float32_fn);
            //y[0] = np.arange(num_class) == y[0];
            //var labels = (np.arange(num_class) == y.reshape(y.shape[0], 1, y.shape[1])).astype(np.float32);
            return (dataset, ToGradient<float>(y));
        }

        static ndarray<T> ToGradient<T>(NDArray data) where T : unmanaged
            => (ndarray<T>)data.ToArray<T>().ToNumPyArray().reshape(data.shape);

        // from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        static void Shuffle<T1, T2>(ndarray<T1> array1, ndarray<T2> array2)
        {
            using var _ = Py.GIL();
            var random = numPy.GetAttr("random");
            var randomState = random.InvokeMethod("get_state");
            random.InvokeMethod("shuffle", array1.PythonObject);
            random.InvokeMethod("set_state", randomState);
            random.InvokeMethod("shuffle", array2.PythonObject);
        }

        static (ndarray<T1>, ndarray<T2>) GetNextBatch<T1, T2>(
            ndarray<T1> inputs, ndarray<T2> outputs, int start, int exclusiveEnd)
        {
            inputs = inputs[start .. (exclusiveEnd-1)];
            outputs = outputs[start .. (exclusiveEnd-1)];
            return (inputs, outputs);
        }

        public void SaveCheckpoint(Session sess)
        {
            var saver = new Saver();
            saver.save(sess, Path.Combine(Name, "mnist_cnn.ckpt"));
        }
    }
}
