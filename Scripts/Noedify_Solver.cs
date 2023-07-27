//#define FIX_0_TRAININGSET
//#define FIX_1_TRAININGSET
//#define CONST_INIT_STATE
//#define TIME_PROFILE
//#define NOEDIFY_BURST
//#define NOEDIFY_MATHEMATICS
//#define NOEDIFY_NORELEASE
//#define NOEDIFY_NOTRAIN
#if NOEDIFY_BURST
using Unity.Burst;
#endif

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;
using Unity.Collections;

public class Noedify_Solver : MonoBehaviour
{
#if NOEDIFY_NORELEASE
    public enum SolverMethod { MainThread, Background, GPU };
#else
    public enum SolverMethod { CPU, GPU };
#endif
    public enum CostFunction { MeanSquare, CrossEntropy };

    public bool trainingInProgress;
    public bool evaluationInProgress;
    public bool suppressMessages = false;
    public float[] cost_report;
    public float[] prediction;
    public float costThreshold = -100f;
    public JobHandle activeJob;
    public DebugReport debug;
    public int fineTuningLayerLimit;

    Coroutine train_coroutine;
    Noedify.Net activeNet;
    List<object> nativeArrayCleanupList;
    List<object> cBufferCleanupList;

    public List<Noedify.Net> nets_cleanup_list;

    public int lastLayer = -1;

    public Noedify_Solver()
    {
        nets_cleanup_list = new List<Noedify.Net>();
    }

    [System.Serializable]
    public class DebugReport
    {
        public bool print_nodeOutputs;
        public bool print_nodeInputs;
        public bool print_weight_Gradients;
        public bool print_bias_Gradients;
        public bool print_deltas;
        public bool print_outputError;

        // public List<string> ReportList;
        // public List<float[]> ReportValues;

        public DebugReport()
        {
            print_nodeOutputs = false;
            print_nodeInputs = false;
            print_weight_Gradients = false;
            print_bias_Gradients = false;
            print_deltas = false;
            print_outputError = false;
            //   ReportList = new List<string>();
            //   ReportValues = new List<float[]>();
        }
#if NOEDIFY_NORELEASE
        public DebugReport_Par ConvertForPar()
        {
            DebugReport_Par debug_par = new DebugReport_Par();
            debug_par.print_nodeOutputs = print_nodeOutputs;
            debug_par.print_nodeInputs = print_nodeInputs;
            debug_par.print_weight_Gradients = print_weight_Gradients;
            debug_par.print_bias_Gradients = print_bias_Gradients;
            debug_par.print_deltas = print_deltas;
            debug_par.print_outputError = print_outputError;
            return debug_par;
        }
#endif
    }
#if NOEDIFY_NORELEASE
    public struct DebugReport_Par
    {
        public bool print_nodeOutputs;
        public bool print_nodeInputs;
        public bool print_weight_Gradients;
        public bool print_bias_Gradients;
        public bool print_deltas;
        public bool print_outputError;

    }
#endif
    public void Evaluate(Noedify.Net net, Noedify.InputArray evaluationInputs, SolverMethod solverMethod = SolverMethod.CPU)
    {
        nets_cleanup_list.Add(net);
        if (debug == null)
            debug = new DebugReport();
        // Check that input shape matches input layer
        if (net.layers[0].layer_type.Equals(Noedify.LayerType.Input))
        {
            if (!evaluationInputs.MatchesSize(net.layers[0].layerSize, net.layers[0].in_channels, suppressMessages))
            {
                if (!suppressMessages) print("ERROR: 1D input wrong size");
                goto Error;
            }
        }
        else if (net.layers[0].layer_type.Equals(Noedify.LayerType.Input2D))
        {
            if (!evaluationInputs.MatchesSize(net.layers[0].layerSize2D, net.layers[0].in_channels, suppressMessages))
            {
                if (!suppressMessages) print("ERROR: 2D input wrong size");
                goto Error;
            }
        }
        else if (net.layers[0].layer_type.Equals(Noedify.LayerType.Input3D))
        {
            if (!evaluationInputs.MatchesSize(net.layers[0].layerSize3D, net.layers[0].in_channels, suppressMessages))
            {
                if (!suppressMessages) print("ERROR: 3D input wrong size");
                goto Error;
            }
        }
        if (solverMethod == SolverMethod.CPU)
        {
            List<float[]> nodeOutputs = new List<float[]>();
            List<float[]> nodeInputs = new List<float[]>();

            ForwardEvaluateNetwork(net, evaluationInputs, ref nodeOutputs, ref nodeInputs);

            if (debug.print_nodeOutputs)
            {
                for (int l = 0; l < net.LayerCount(); l++)
                {
                    string layerOutput = "layer " + l + " node outputs: ";
                    for (int j = 0; j < net.layers[l].layerSize; j++)
                        layerOutput += nodeOutputs[l][j] + ", ";
                    print(layerOutput);
                }
            }
            if (debug.print_nodeInputs)
            {
                for (int l = 0; l < net.LayerCount(); l++)
                {
                    string layerOutput = "layer " + l + " node inputs: ";
                    for (int j = 0; j < net.layers[l].layerSize; j++)
                        layerOutput += nodeInputs[l][j] + ", ";
                    print(layerOutput);
                }
            }

            if (lastLayer < 0)
            {
                prediction = nodeOutputs[net.LayerCount() - 1];
            }
            else
            {
                prediction = nodeOutputs[lastLayer];
                Debug.Log("Stopping evaluation at layer " + lastLayer + " (" + net.layers[lastLayer].name + ")");
            }

            return;
        }
#if NOEDIFY_NORELEASE
        else if (solverMethod == SolverMethod.Background)
        {
            activeNet = net;
            evaluationInProgress = true;
            if (net.trainingInProgress == false)
            {
                if (!activeNet.nativeArraysInitialized)
                    net.GenerateNativeParameterArrays();

                train_coroutine = StartCoroutine(Evaluate_Par_Queue(net, evaluationInputs));
                return;
            }
            else
            {
                if (!suppressMessages) print("Error: Evaluation initiated before previous job competed");
                return;
            }
        }
#endif
        else if (solverMethod == SolverMethod.GPU)
        {

            if (net._shader == null)
            {
                Debug.LogError("(Noedify_Solver.Evaluate_CBuff): net._shader not assigned");
                goto Error;
            }

            ComputeShader _shader = net._shader;
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            if (!net.nativeArraysInitialized)
                net.GenerateNativeParameterArrays();
            if (!net.cBuffersInitialized)
                net.GenerateComputeBuffers();
            sw.Stop();
            //print("GPU initialization time: " + sw.ElapsedMilliseconds + " ms");
            sw.Reset(); sw.Start();
            Evaluate_CBuff(net, evaluationInputs, _shader);
            sw.Stop();
            //print("Evaluation time: " + sw.ElapsedMilliseconds + " ms");
            return;
        }
        Error:
        print("Error");
        return;
    }

#if NOEDIFY_NOTRAIN
    public void TrainNetwork(Noedify.Net net, List<Noedify.InputArray> trainingInputs, List<float[]> trainingOutputs, int no_epochs, int batch_size, float trainingRate, CostFunction costFunction, SolverMethod solverMethod, List<float> trainingSetWeighting = null, int N_threads = 8)
    {
        nets_cleanup_list.Add(net);
        activeNet = net;
#if (CONST_INIT_STATE)
        Random.InitState(5);
#endif
        if (debug == null)
            debug = new DebugReport();
        if (trainingSetWeighting == null)
        {
            trainingSetWeighting = new List<float>();
            for (int n = 0; n < trainingInputs.Count; n++)
                trainingSetWeighting.Add(1f);
        }
        else if (trainingSetWeighting.Count != trainingInputs.Count)
        {
            if (!suppressMessages) print("Error: Number of training weights (" + trainingSetWeighting.Count + ") doesn't match number of training input sets (" + trainingInputs.Count + ")");
            if (!suppressMessages) print("Warning: Ignoring trainingSetWeighting.");
            trainingSetWeighting = new List<float>();
            for (int n = 0; n < trainingInputs.Count; n++)
                trainingSetWeighting.Add(1f);
        }
        else
        {
            for (int n = 0; n < trainingSetWeighting.Count; n++)
            {
                if (trainingSetWeighting[n] < -1f)
                    trainingSetWeighting[n] = -1f;
                else if (trainingSetWeighting[n] > 1f)
                    trainingSetWeighting[n] = 1f;
            }
        }

        float initialTrainingRate = trainingRate;


        if (solverMethod == SolverMethod.CPU)
        {
            int no_training_sets = trainingInputs.Count;
            if (trainingInputs.Count != trainingOutputs.Count)
                if (!suppressMessages) print("Error: Number of training input arrays (" + trainingInputs.Count + ") doesn't match number of training output arrays (" + trainingOutputs.Count + ")");
            cost_report = new float[no_epochs];

            if (batch_size > no_training_sets)
            {
                batch_size = no_training_sets;
                if (!suppressMessages) print("WARNING: batch size greater than number of training sets. batch_size reduced to " + no_training_sets);
            }

            if (batch_size <= no_training_sets)
            {
                if (!suppressMessages) print("Starting training for " + no_epochs + " epochs with batch size " + batch_size + " across " + no_training_sets + " training sets");
                for (int epoch = 0; epoch < no_epochs; epoch++)
                {
                    //print("epoch " + epoch + " of " + no_epochs);
                    int[] trainingSets = new int[no_training_sets];
                    for (int i = 0; i < no_training_sets; i++)
                        trainingSets[i] = i;
                    trainingSets = Noedify_Utils.Shuffle(trainingSets);
#if FIX_0_TRAININGSET
                trainingSets[0] = 0;
#endif
#if FIX_1_TRAININGSET
                trainingSets[1] = 6;
#endif
                    List<Noedify.NN_LayerGradients[]> batchGradients = new List<Noedify.NN_LayerGradients[]>();
                    float cost = 0;
                    for (int batch = 0; batch < batch_size; batch++)
                    {
                        int set_no = trainingSets[batch];

                        List<float[]> nodeInputs = new List<float[]>();
                        List<float[]> nodeOutputs = new List<float[]>();
                        ForwardEvaluateNetwork(net, trainingInputs[set_no], ref nodeOutputs, ref nodeInputs);
                        batchGradients.Add(BackpropagateNetwork(net, costFunction, trainingOutputs[set_no], nodeOutputs, nodeInputs));
                        cost += Cost(nodeOutputs[net.LayerCount() - 1], trainingOutputs[set_no], costFunction) / batch_size;

                        if (debug.print_nodeOutputs)
                        {
                            for (int l = 0; l < net.LayerCount(); l++)
                            {
                                string layerOutput = "Epoch " + epoch + ", layer " + l + " node outputs: ";
                                for (int j = 0; j < net.layers[l].layerSize; j++)
                                    layerOutput += "(" + j + ")" + nodeOutputs[l][j] + ", ";
                                print(layerOutput);
                            }
                            print("------------------------------------");
                        }
                        if (debug.print_nodeInputs)
                        {
                            for (int l = 0; l < net.LayerCount(); l++)
                            {
                                string layerOutput = "layer " + l + " node inputs: ";
                                for (int j = 0; j < net.layers[l].layerSize; j++)
                                    layerOutput += nodeInputs[l][j] + ", ";
                                print(layerOutput);
                            }
                        }


                    }
                    if ((!suppressMessages) & (epoch % 10 == 0 | epoch == no_epochs - 1))
                        print("epoch " + (epoch + 1) + "/" + no_epochs + ", cost: " + cost);
                    cost_report[epoch] = cost;
                    if (epoch > 0)
                    {
                        if (Mathf.Clamp(cost_report[epoch - 1] - cost_report[epoch] / cost_report[epoch - 1], 0, 1f) > .2f)
                        {
                            if (trainingRate > (initialTrainingRate / 10f))
                            {
                                //net.trainingRate /= 2;
                                //print("new training rate: " + net.trainingRate);
                            }
                        }
                    }

                    float[] weighting = new float[batch_size];
                    for (int batch = 0; batch < batch_size; batch++)
                        weighting[batch] = trainingSetWeighting[trainingSets[batch]];
                    Noedify.NN_LayerGradients[] netGradients = AccumulateGradients(net, trainingRate, batchGradients, batch_size, weighting);
                    if (debug.print_bias_Gradients & ((epoch % 10 == 0) || (epoch == no_epochs - 1)))
                    {
                        for (int l = 1; l < net.LayerCount(); l++)
                        {
                            string biasGradientString = "Epoch " + epoch + " Accumulated: bias gradients layer " + l + ": ";
                            if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
                            {
                                for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                                    biasGradientString += "(b" + f + "=" + net.layers[l].biases.valuesConv[f] + ")" + netGradients[l - 1].bias_gradients.valuesConv[f] + " ";
                            }
                            else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D) { }
                            else
                            {
                                for (int j = 0; j < net.layers[l].layerSize; j++)
                                    biasGradientString += "(b" + j + "=" + net.layers[l].biases.values[j] + ")" + netGradients[l - 1].bias_gradients.values[j] + " ";
                            }
                            print(biasGradientString);
                        }
                    }
                    if (debug.print_weight_Gradients & ((epoch % 10 == 0) || (epoch == no_epochs - 1)))
                    {
                        for (int l = 1; l < net.LayerCount(); l++)
                        {
                            if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
                            {
                                for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                                {
                                    string weightGradientsDebug = "Epoch " + epoch + " layer " + l + " from filter " + f + ": ";

                                    for (int j = 0; j < net.layers[l].conv2DLayer.N_weights_per_filter; j++)
                                    {
                                        weightGradientsDebug += "(w_" + f + "_" + j + "=" + net.layers[l].weights.valuesConv[f, j] + ")" + netGradients[l - 1].weight_gradients.valuesConv[f, j] + " ";
                                    }
                                    print(weightGradientsDebug);
                                }
                            }
                            else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D) { }
                            else
                            {
                                for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                                {
                                    if (i > 10)
                                        break;
                                    string weightGradientsDebug = "Epoch " + epoch + " layer " + l + " from node " + i + ": ";
                                    for (int j = 0; j < net.layers[l].layerSize; j++)
                                    {
                                        weightGradientsDebug += "(w_" + i + "_" + j + "=" + net.layers[l].weights.values[i, j] + ")" + netGradients[l - 1].weight_gradients.values[i, j] + " ";
                                    }
                                    print(weightGradientsDebug);
                                }
                            }
                        }
                    }
                    net.ApplyGradients(netGradients, batch_size, false, fineTuningLayerLimit);
                    if (cost < costThreshold)
                    {
                        if (!suppressMessages) print("Cost Threshold reached (" + costThreshold + "). Training Complete.");
                        break;
                    }
                }
            }
            return;
        }
        else
        {
            trainingInProgress = true;
            if (net.trainingInProgress == false)
            {
                int no_training_sets = trainingInputs.Count;
                if (trainingInputs.Count != trainingOutputs.Count)
                {
                    if (!suppressMessages) print("Error: Number of training input arrays (" + trainingInputs.Count + ") doesn't match number of training output arrays (" + trainingOutputs.Count + ")");
                    return;
                }
                cost_report = new float[no_epochs];

                if (batch_size > no_training_sets)
                {
                    batch_size = no_training_sets;
                    if (!suppressMessages) print("WARNING: batch size greater than number of training sets. batch_size reduced to " + no_training_sets);
                }
                net.GenerateNativeParameterArrays();
                train_coroutine = StartCoroutine(TrainNetwork_Par_Queue(net, batch_size, no_training_sets, no_epochs, trainingRate, costFunction, trainingInputs, trainingOutputs, trainingSetWeighting, N_threads));
                return;
            }
            else
            {
                if (!suppressMessages) print("Error: Training initiated before previous job competed");
                return;
            }
        }
    }
#endif

    void ForwardEvaluateNetwork(Noedify.Net net, Noedify.InputArray trainingInputs, ref List<float[]> nodeOutputs, ref List<float[]> nodeInputs)
    {

        nodeOutputs.Add(trainingInputs.FlattenArray());
        nodeInputs.Add(new float[net.layers[0].layerSize]);

        int lastEvalLyr = net.LayerCount();
        if (lastLayer >= 0)
            lastEvalLyr = lastLayer + 1;

        for (int l = 1; l < lastEvalLyr; l++)
        {
            //print("evaluating layer " + l + " (" + net.layers[l].name + ")");
            float[] layerInput = nodeOutputs[l - 1];
            float[] layerNodeOutputs = new float[net.layers[l].layerSize];
            float[] layerNodeInputs = new float[net.layers[l].layerSize];
            EvaluateLayer(net, net.layers[l], layerInput, ref layerNodeOutputs, ref layerNodeInputs);
            nodeOutputs.Add(layerNodeOutputs);
            nodeInputs.Add(layerNodeInputs);
            //Noedify_Utils.PrintArray<float>(layerNodeOutputs, "out");
        }
    }

    void EvaluateLayer(Noedify.Net net, Noedify.Layer layer, float[] inputs, ref float[] nodeOutputs, ref float[] nodeInputs)
    {
        //Debug.Log("evaluating layer: " + layer.layer_no);
        switch (layer.layer_type)
        {
            case (Noedify.LayerType.Convolutional2D):
                {
                    for (int j = 0; j < layer.layerSize; j++)
                        nodeInputs[j] = layer.biases.valuesConv[layer.conv2DLayer.filterTrack[j]];
                    for (int j = 0; j < layer.layerSize; j++)
                    {
                        for (int conn = 0; conn < layer.conv2DLayer.N_connections_per_node; conn++)
                        {
                            int connected_i = layer.conv2DLayer.connections[layer.conv2DLayer.nodeTrack[j], conn];
                            if (connected_i != -1)
                            {
                                int nif = layer.conv2DLayer.filterTrack[j];
                                int cif = layer.conv2DLayer.connectionsInFilter[layer.conv2DLayer.nodeTrack[j], conn];
                                //float og = nodeInputs[j];
                                nodeInputs[j] += inputs[connected_i] * layer.weights.valuesConv[nif, cif];
                                //nodeInputs[i] += inputs[connected_j];
                                //print("node " + (layer.conv2DLayer.nodeTrack[j], conn) + " ch " + nif + ": " + og + " + " + inputs[connected_i] + " * " + layer.weights.valuesConv[nif, cif] + "(weight[" + nif + "," + cif + "]) = " + nodeInputs[j]);
                            }
                        }
                    }

                    break;
                }
            case (Noedify.LayerType.Convolutional3D):
                {

                    for (int j = 0; j < layer.layerSize; j++)
                        nodeInputs[j] = layer.biases.valuesConv[layer.conv3DLayer.filterTrack[j]];
                    for (int j = 0; j < layer.layerSize; j++) // for each node in current layer
                    {
                        // Note: N_connections_per_node = filterSize * prev_lyr_channels
                        for (int conn = 0; conn < layer.conv3DLayer.N_connections_per_node; conn++) // for each connection to current node
                        {

                            int connected_i = layer.conv3DLayer.connections[layer.conv3DLayer.nodeTrack[j], conn];
                           // if (j == 0)
                                //print("node j=0 connection " + conn + ": " + connected_i);
                            if (connected_i != -1)
                            {
                                int nif = layer.conv3DLayer.filterTrack[j];
                                int cif = layer.conv3DLayer.connectionsInFilter[layer.conv3DLayer.nodeTrack[j], conn];
                                //float og = nodeInputs[j];

                                nodeInputs[j] += inputs[connected_i] * layer.weights.valuesConv[nif, cif];
                                //print("node " + (layer.conv3DLayer.nodeTrack[j], conn) + " ch " + nif + ": " + og + " + " + inputs[connected_i] + "(connectedi: " + connected_i + ") * " + layer.weights.valuesConv[nif, cif] + "(weight[" + nif + "," + cif + "]) = " + nodeInputs[j]);
                            }
                        }
                    }
                    break;
                }
            case (Noedify.LayerType.Pool2D):
                {
                    for (int j = 0; j < layer.layerSize; j++)
                    {
                        if (layer.pool_type == Noedify.PoolingType.Max)
                            nodeInputs[j] = -100;
                        else if (layer.pool_type == Noedify.PoolingType.Avg)
                            nodeInputs[j] = 0;
                        int connsMade = 0;
                        for (int i = 0; i < inputs.Length; i++)
                        {
                            if (layer.conv2DLayer.connectionMask[i, j] >= 0)
                            {
                                connsMade++;
                                if (layer.pool_type == Noedify.PoolingType.Avg)
                                    nodeInputs[j] += inputs[i];
                                else if (layer.pool_type == Noedify.PoolingType.Max)
                                {
                                    if (inputs[i] > nodeInputs[j])
                                    {
                                        nodeInputs[j] = inputs[i];
                                    }
                                }
                            }
                            if (connsMade >= layer.conv2DLayer.N_connections_per_node)
                            {
                                break;
                            }
                        }

                        if (layer.pool_type == Noedify.PoolingType.Avg)
                            nodeInputs[j] /= layer.conv2DLayer.filterSize[0] * layer.conv2DLayer.filterSize[1];
                    }
                    break;
                }
            case (Noedify.LayerType.TranspConvolutional2D):
                {
                    if (Noedify_Utils.Is1DLayerType(net.layers[layer.layer_no - 1])) // 1D layer to Transp2DConvolutional
                    {
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            nodeInputs[j] = 0;
                            for (int i = 0; i < inputs.Length; i++)
                                nodeInputs[j] += inputs[i] * layer.weights.values[i, j];
                            nodeInputs[j] += layer.biases.valuesConv[layer.conv2DLayer.channelTrack[j]];
                        }
                    }
                    else // 2D layer to Transp2DConvolutional
                    {

                        for (int j = 0; j < layer.layerSize; j++)
                            nodeInputs[j] = layer.biases.valuesConv[layer.conv2DLayer.filterTrack[j]];
                        // For each node in current layer:
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            int prevLyrSize = net.layers[layer.layer_no - 1].layerSize2D[0] * net.layers[layer.layer_no - 1].layerSize2D[1];
                            int prevkernelSize = net.layers[layer.layer_no - 1].conv2DLayer.filterSize[0] * net.layers[layer.layer_no - 1].conv2DLayer.filterSize[1];
                            int kernelSize = layer.conv2DLayer.filterSize[0] * layer.conv2DLayer.filterSize[1];
                            // For each connection from node j to previous layer:
                            for (int conn = 0; conn < layer.conv2DLayer.N_connections_per_node; conn++)
                            {
                                int nif_j = layer.conv2DLayer.nodeTrack[j];
                                int connected_nif_i = layer.conv2DLayer.connections[nif_j, conn];
                                // If connection exists:
                                if (connected_nif_i > -1)
                                {
                                    // Connections are mirrored across previous filters. For each mirrored filter in previous filter (f_last):
                                    for (int f_last = 0; f_last < net.layers[layer.layer_no - 1].conv2DLayer.no_filters; f_last++)
                                    {
                                        int connected_i = connected_nif_i + prevLyrSize * f_last;
                                        int f_j = layer.conv2DLayer.filterTrack[j];
                                        int cif = layer.conv2DLayer.connectionsInFilter[nif_j, conn] + kernelSize * f_last;
                                        float inp = inputs[connected_i];
                                        float wght = layer.weights.valuesConv[f_j, cif];
                                        nodeInputs[j] += inp * wght;

                                    }
                                }
                            }
                        }


                    }
                    break;
                }
            case (Noedify.LayerType.TranspConvolutional3D):
                {
                    if (Noedify_Utils.Is1DLayerType(net.layers[layer.layer_no - 1])) // 1D layer to Transp3DConvolutional
                    {
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            nodeInputs[j] = 0;
                            for (int i = 0; i < inputs.Length; i++)
                                nodeInputs[j] += inputs[i] * layer.weights.values[i, j];
                            nodeInputs[j] += layer.biases.valuesConv[layer.conv3DLayer.channelTrack[j]];
                        }
                    }
                    else // 3D layer to Transp3DConvolutional
                    {

                        for (int j = 0; j < layer.layerSize; j++)
                            nodeInputs[j] = layer.biases.valuesConv[layer.conv3DLayer.filterTrack[j]];
                        // For each node in current layer:
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            int j_conn_count = 0;
                            int prevLyrSize = net.layers[layer.layer_no - 1].layerSize3D[0] * net.layers[layer.layer_no - 1].layerSize3D[1] * net.layers[layer.layer_no - 1].layerSize3D[2];
                            int prevkernelSize = net.layers[layer.layer_no - 1].conv3DLayer.filterSize[0] * net.layers[layer.layer_no - 1].conv3DLayer.filterSize[1] * net.layers[layer.layer_no - 1].conv3DLayer.filterSize[2];
                            int kernelSize = layer.conv3DLayer.filterSize[0] * layer.conv3DLayer.filterSize[1] * layer.conv3DLayer.filterSize[2];
                            // For each connection from node j to previous layer:
                            for (int conn = 0; conn < layer.conv3DLayer.N_connections_per_node; conn++)
                            {
                                int nif_j = layer.conv3DLayer.nodeTrack[j];
                                int connected_nif_i = layer.conv3DLayer.connections[nif_j, conn];
                                //print("j=" + j + ": connections[" + nif_j + ", " + conn + "] = " + connected_nif_i);
                                // If connection exists:
                                if (connected_nif_i > -1)
                                {
                                    j_conn_count++;
                                    // Connections are mirrored across previous filters. For each mirrored filter in previous filter (f_last):
                                    for (int f_last = 0; f_last < net.layers[layer.layer_no - 1].conv3DLayer.no_filters; f_last++)
                                    {
                                        int connected_i = connected_nif_i + prevLyrSize * f_last;
                                        int f_j = layer.conv3DLayer.filterTrack[j];
                                        int cif = layer.conv3DLayer.connectionsInFilter[nif_j, conn] + kernelSize * f_last;
                                        float inp = inputs[connected_i];
                                        //print("conn#: " + conn + ", j=" + j + ", f_j=" + f_j + ", connected_i=" + connected_i + ", cif=" + cif);
                                        //print("weights dims: [" + layer.weights.valuesConv.GetLength(0) + ", " + layer.weights.valuesConv.GetLength(1) + "]");
                                        float wght = layer.weights.valuesConv[f_j, cif];

                                        nodeInputs[j] += inp * wght;

                                    }
                                }
                            }
                            //print("j=" + j + " #of connections: " + j_conn_count);
                        }


                    }
                    break;
                }
            case (Noedify.LayerType.BatchNorm2D):
                {
                    float[] filterMean = new float[layer.in_channels];
                    float[] filterVar = new float[layer.in_channels];
                    if (layer.bn_running_track){
                        filterMean = (float[])layer.bn_running_mean.Clone();
                        filterVar = (float[])layer.bn_running_var.Clone();
                    }
                    else
                    {
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            //print("filterMean[" + layer.conv2DLayer.filterTrack[j] + "](j=" + j + ") += " + inputs[j] + " = " + (filterMean[layer.conv2DLayer.filterTrack[j]] + inputs[j]));
                            filterMean[layer.conv2DLayer.filterTrack[j]] += inputs[j];
                        }
                                            //print("filtermean: " + filterMean[0] + " / " + (layer.layerSize2D[0] * layer.layerSize2D[1]) + " + " + filterMean[0]);

                        for (int c = 0; c < layer.in_channels; c++)
                            filterMean[c] /= (layer.layerSize2D[0] * layer.layerSize2D[1]);
                        

                        
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            int filter_no = layer.conv2DLayer.filterTrack[j];
                            float varTemp = (inputs[j] - filterMean[filter_no]);
                            filterVar[filter_no] += varTemp * varTemp;
                        }
                        for (int c = 0; c < layer.in_channels; c++)
                            filterVar[c] = filterVar[c] / ((layer.layerSize2D[0] * layer.layerSize2D[1]));
                    }
                    /*
                    string outString = "filterMean: ";
                    for (int c = 0; c < layer.in_channels; c++)
                        outString += filterMean[c] + ", ";
                    print(outString);
                    string outStringVar = "filterVar: ";
                    for (int c = 0; c < layer.in_channels; c++)
                        outStringVar += filterVar[c] + ", ";
                    print(outStringVar);
                    */
                    for (int c = 0; c < layer.in_channels; c++)
                        filterVar[c] = Mathf.Sqrt(filterVar[c] + layer.bn_eps);

                    for (int j = 0; j < layer.layerSize; j++)
                    {
                        int filter_no = layer.conv2DLayer.filterTrack[j];
                        float inputHat = (inputs[j] - filterMean[filter_no]) / filterVar[filter_no];
                        nodeInputs[j] = layer.weights.values[0, filter_no] * inputHat + layer.biases.values[filter_no];

                    }

                    break;
                }
            case (Noedify.LayerType.BatchNorm3D):
                {
                    float[] filterMean = new float[layer.in_channels];
                    float[] filterVar = new float[layer.in_channels];
                    if (layer.bn_running_track){
                        filterMean = layer.bn_running_mean;
                        filterVar = layer.bn_running_var;
                    }
                    else{
                    for (int j = 0; j < layer.layerSize; j++)
                    {
                        //print("filterMean[" + layer.conv3DLayer.filterTrack[j] + "](j=" + j + ") += " + inputs[j] + " = " + (filterMean[layer.conv3DLayer.filterTrack[j]] + inputs[j]));
                        filterMean[layer.conv3DLayer.filterTrack[j]] += inputs[j];
                    }

                        //print("filtermean: " + filterMean[0] + " / " + (layer.layerSize3D[0] * layer.layerSize3D[1]) + " + " + filterMean[0]);

                        for (int c = 0; c < layer.in_channels; c++)
                            filterMean[c] /= (layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]);
                        /*
                        string outString = "filterMean: ";
                        for (int c = 0; c < layer.in_channels; c++)
                            outString += filterMean[c] + ", ";
                        print(outString);
                        */
                        for (int j = 0; j < layer.layerSize; j++)
                        {
                            int filter_no = layer.conv3DLayer.filterTrack[j];
                            float varTemp = (inputs[j] - filterMean[filter_no]);
                            filterVar[filter_no] += varTemp * varTemp;
                        }
                        /*
                        string outStringVar = "filterVar: ";
                        for (int c = 0; c < layer.in_channels; c++)
                            outStringVar += filterVar[c] + ", ";
                        print(outStringVar);
                        */
                        for (int c = 0; c < layer.in_channels; c++)
                            filterVar[c] = filterVar[c] / ((layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]));
                    }
                    
                    for (int c = 0; c < layer.in_channels; c++)
                        filterVar[c] = Mathf.Sqrt(filterVar[c] + layer.bn_eps);

                    for (int j = 0; j < layer.layerSize; j++)
                    {
                        int filter_no = layer.conv3DLayer.filterTrack[j];
                        float inputHat = (inputs[j] - filterMean[filter_no]) / filterVar[filter_no];
                        nodeInputs[j] = layer.weights.values[0, filter_no] * inputHat + layer.biases.values[filter_no];
                    }
                    break;
                }
            case (Noedify.LayerType.FullyConnected):
            case (Noedify.LayerType.Output):
                {
                    for (int j = 0; j < layer.layerSize; j++)
                    {
                        nodeInputs[j] = 0;
                        for (int i = 0; i < inputs.Length; i++)
                        {
                            nodeInputs[j] += inputs[i] * layer.weights.values[i, j];
                        }
                        nodeInputs[j] += layer.biases.values[j];
                    }
                    break;
                }
            default: print("Error (Noedify_Solver.EvaluateLayer): Unknown layer type " + layer.layer_type.ToString()); break;
        }
        nodeOutputs = ApplyActivationFunction(nodeInputs, layer.activationFunction);
    }

#if NOEDIFY_NOTRAIN
    Noedify.NN_LayerGradients[] BackpropagateNetwork(Noedify.Net net, CostFunction costFunction, float[] trainingOutputs, List<float[]> layerNodeOutputs, List<float[]> layerNodeInputs)
    {
        float[] deltas = DeltaCost(layerNodeOutputs[net.LayerCount() - 1], trainingOutputs, costFunction);

        if (debug.print_outputError)
        {
            string errorCalcString = "Cost Calculation: ";
            int output_layer_no = net.LayerCount() - 1;
            for (int j = 0; j < net.layers[output_layer_no].layerSize; j++)
            {
                float nodeOut = layerNodeOutputs[output_layer_no][j];
                float trainOut = trainingOutputs[j];
                float delTest = deltas[j];
                errorCalcString += "(" + j + ")[" + nodeOut + " - " + trainOut + " = " + delTest + "] , ";
            }
            print(errorCalcString);
        }
        float[] delta_z = new float[net.layers[net.LayerCount() - 1].layerSize];
        delta_z = ApplyDeltaActivationFunction(layerNodeInputs[net.LayerCount() - 1], net.layers[net.LayerCount() - 1].activationFunction);
        for (int i = 0; i < net.layers[net.LayerCount() - 1].layerSize; i++)
            deltas[i] *= delta_z[i];
        if (debug.print_deltas)
        {
            string outStringDelta = "Output layer deltas: ";
            for (int i = 0; i < net.layers[net.LayerCount() - 1].layerSize; i++)
                outStringDelta += "(" + delta_z[i] + ")" + deltas[i] + ", ";
            print(outStringDelta);
        }
        // Calculate gradients for output layer
        Noedify.NN_LayerGradients[] gradients = new Noedify.NN_LayerGradients[net.LayerCount() - 1];
        gradients[net.LayerCount() - 2] = CalculateGradientsLayer(net, net.layers[net.LayerCount() - 1], layerNodeOutputs[net.LayerCount() - 2], deltas);
        // Backpropagate through network
        for (int l = (net.LayerCount() - 2); l > 0; l--)
        {
            deltas = BackpropagateLayer(net, net.layers[l], net.layers[l + 1], deltas, layerNodeInputs[l], layerNodeOutputs[l], layerNodeInputs[l + 1]);
            gradients[l - 1] = CalculateGradientsLayer(net, net.layers[l], layerNodeOutputs[l - 1], deltas);

            if (debug.print_deltas)
            {
                string outStringDelta = "layer " + l + " deltas: ";
                for (int i = 0; i < net.layers[l].layerSize; i++)
                    outStringDelta += deltas[i] + ", ";
                print(outStringDelta);
            }
            if (l == fineTuningLayerLimit)
                break;
        }
        return gradients;
    }

    Noedify.NN_LayerGradients CalculateGradientsLayer(Noedify.Net net, Noedify.Layer layer, float[] layerOutputs_prevLyr, float[] deltas)
    {
        Noedify.NN_LayerGradients gradients = new Noedify.NN_LayerGradients(net.layers, layer.layer_no);
        if (layer.layer_type == Noedify.LayerType.Pool2D)
            return gradients;
        for (int j = 0; j < layer.layerSize; j++)
        {
            int connsMade = 0;
            //print("calculating weight gradients");
            for (int i = 0; i < layerOutputs_prevLyr.Length; i++)
            {
                if (layer.layer_type == Noedify.LayerType.Convolutional2D)
                {
                    if (layer.conv2DLayer.connectionMask[i, j] >= 0)
                    {
                        connsMade++;
                        gradients.weight_gradients.valuesConv[layer.conv2DLayer.filterTrack[j], layer.conv2DLayer.connectionMask[i, j]] += deltas[j] * layerOutputs_prevLyr[i];
                    }
                }
                else
                    gradients.weight_gradients.values[i, j] += deltas[j] * layerOutputs_prevLyr[i];
                if (layer.layer_type == Noedify.LayerType.Convolutional2D)
                    if (connsMade >= layer.conv2DLayer.N_connections_per_node)
                        break;
            }
            //print("calculating bias gradients");
            if (layer.layer_type == Noedify.LayerType.Convolutional2D)
            {
                gradients.bias_gradients.valuesConv[layer.conv2DLayer.filterTrack[j]] += deltas[j];
            }
            else
                gradients.bias_gradients.values[j] = deltas[j];
        }
        return gradients;
    }

    float[] BackpropagateLayer(Noedify.Net net, Noedify.Layer prevLayer, Noedify.Layer nextLayer, float[] deltas, float[] layerNodeInputs, float[] layerNodeOutputs, float[] nextLayerNodeInputs)
    {

        float[] deltas_new = new float[prevLayer.layerSize];
        //print("Backpropagating from " + nextLayer.name + " to " + prevLayer.name);
        for (int i = 0; i < prevLayer.layerSize; i++)
        {
            deltas_new[i] = 0;
            for (int j = 0; j < deltas.Length; j++)
            {
                if (nextLayer.layer_type == Noedify.LayerType.Convolutional2D)
                {
                    if (nextLayer.conv2DLayer.connectionMask[i, j] >= 0)
                    {
                        deltas_new[i] += nextLayer.weights.valuesConv[nextLayer.conv2DLayer.filterTrack[j], nextLayer.conv2DLayer.connectionMask[i, j]] * deltas[j];
                    }
                }
                else if (nextLayer.layer_type == Noedify.LayerType.Pool2D)
                {
                    if (nextLayer.conv2DLayer.connectionMask[i, j] >= 0)
                    {
                        if (nextLayer.pool_type == Noedify.PoolingType.Avg)
                            deltas_new[i] += deltas[j] / nextLayer.conv2DLayer.filterSize[0] / nextLayer.conv2DLayer.filterSize[1];
                        else if (nextLayer.pool_type == Noedify.PoolingType.Max)
                            if (layerNodeOutputs[i] == nextLayerNodeInputs[j])
                            {
                                deltas_new[i] += deltas[j];
                            }
                    }
                }
                else
                    deltas_new[i] += nextLayer.weights.values[i, j] * deltas[j];
            }
            //deltas_new[i] *= ApplyDeltaActivationFunction(layerNodeInputs[i], prevLayer.activationFunction);
        }
        if (prevLayer.layer_type == Noedify.LayerType.Convolutional2D)
        {
            float[] delta_z = ApplyDeltaActivationFunction(layerNodeInputs, prevLayer.activationFunction);
            for (int i = 0; i < prevLayer.layerSize; i++) // *********************** //
                deltas_new[i] *= delta_z[i];
        }

        return deltas_new;
    }
#endif
#if NOEDIFY_NOTRAIN
    IEnumerator Evaluate_Par_Queue(Noedify.Net net, Noedify.InputArray inputs)
    {
        if (net.trainingInProgress == false)
        {
            CleanupTrainingNativeArrays();
            NativeArray<Noedify.ActivationFunction> activationFunctionList = new NativeArray<Noedify.ActivationFunction>(net.LayerCount(), Allocator.Persistent);
            NativeArray<Noedify.PoolingType> poolingTypeList = new NativeArray<Noedify.PoolingType>(net.LayerCount(), Allocator.Persistent);
            NativeArray<Noedify.LayerType> layerTypeList = new NativeArray<Noedify.LayerType>(net.LayerCount(), Allocator.Persistent);
            nativeArrayCleanupList.Add(activationFunctionList);
            nativeArrayCleanupList.Add(poolingTypeList);
            nativeArrayCleanupList.Add(layerTypeList);
            NativeArray<int> layerSizeList = NativeAllocInt(net.LayerCount());
            NativeArray<float> netInputs = NativeAllocFloat(net.layers[0].layerSize);
            NativeArray<float> netOutputs = NativeAllocFloat(net.layers[net.LayerCount() - 1].layerSize);
            NativeArray<float> nodeOutputs_par = NativeAllocFloat(net.total_no_nodes);
            NativeArray<float> nodeInputs_par = NativeAllocFloat(net.total_no_nodes);
            NativeArray<int> CNN_nodes_per_filter_par = NativeAllocInt(net.LayerCount() - 1);
            NativeArray<int> CNN_weights_per_filter_par = NativeAllocInt(net.LayerCount() - 1);
            NativeArray<int> CNN_no_filters_par = NativeAllocInt(net.LayerCount() - 1);
            NativeArray<int> CNN_no_conns_per_node = NativeAllocInt(net.LayerCount() - 1);

            InitializeNetTempNativeArrays(
                net,
                ref activationFunctionList,
                ref layerTypeList,
                ref layerSizeList,
                ref CNN_nodes_per_filter_par,
                ref CNN_weights_per_filter_par,
                ref CNN_no_filters_par,
                ref CNN_no_conns_per_node,
                ref poolingTypeList);

            try
            {
                for (int i = 0; i < net.layers[0].layerSize; i++)
                    netInputs[i] = inputs.FlattenArray()[i];
            }
            catch
            {
                print("ERROR (Noedify_Solver.Evaluate_Par): Node input size (" + inputs.w + ") incompatable with input layer size (" + net.layers[0].layerSize + ")");
                goto Error;
            }

            EvaluateNetwork_Job evalNetwork_Job = new EvaluateNetwork_Job();

            evalNetwork_Job.activationFunction = activationFunctionList;
            evalNetwork_Job.layerTypes = layerTypeList;
            evalNetwork_Job.layerPooling = poolingTypeList;
            evalNetwork_Job.layerSize = layerSizeList;
            evalNetwork_Job.networkWeights = net.networkWeights_par;
            evalNetwork_Job.networkBiases = net.networkBiases_par;
            evalNetwork_Job.weightIndeces_start = net.weightIdx_start;
            evalNetwork_Job.biasIndeces_start = net.biasIdx_start;
            evalNetwork_Job.activeNodeIndeces_start = net.activeNodeIdx_start;
            evalNetwork_Job.nodeIndeces_start = net.nodeIdx_start;
            evalNetwork_Job.nodeInputs = nodeInputs_par;
            evalNetwork_Job.nodeOutputs = nodeOutputs_par;
            evalNetwork_Job.connectionMask = net.connectionMask_par;
            evalNetwork_Job.connectionMaskStart_index = net.connectionMaskIndeces_start;
            evalNetwork_Job.CNN_nodes_per_filter = CNN_nodes_per_filter_par;
            evalNetwork_Job.CNN_weights_per_filter = CNN_weights_per_filter_par;
            evalNetwork_Job.CNN_no_filters = CNN_no_filters_par;
            evalNetwork_Job.CNN_conns_per_node = CNN_no_conns_per_node;
            evalNetwork_Job.networkInputs = netInputs;
            evalNetwork_Job.networkOutputs = netOutputs;
            evalNetwork_Job.total_no_active_nodes = net.total_no_activeNodes;
            evalNetwork_Job.network_input_size = net.layers[0].layerSize;

            JobHandle eval_handle;
            eval_handle = evalNetwork_Job.Schedule(1, 1);
            activeJob = eval_handle;
            net.trainingInProgress = true;

            int timeoutCounter = 0;
            while (true)
            {
                yield return null;
                timeoutCounter++;
                if (eval_handle.IsCompleted)
                {
                    break;
                }
                else if (timeoutCounter > 100000)
                {
                    if (!suppressMessages) print("Error: Evaluation timeout reached!");
                    goto Error;
                }
            }
            eval_handle.Complete();



            Error:
            prediction = netOutputs.ToArray();
            CleanupTrainingNativeArrays();
            net.trainingInProgress = false;
            evaluationInProgress = false;
        }
        else
        {
            if (!suppressMessages) print("Error: Evaluation initiated before previous job competed");
            evaluationInProgress = false;
            net.Cleanup_Par();
        }
        net.trainingInProgress = false;
        yield return null;
    }

    IEnumerator TrainNetwork_Par_Queue(Noedify.Net net, int batch_size, int no_training_sets, int no_epochs, float trainingRate, CostFunction costFunction, List<Noedify.InputArray> trainingInputs, List<float[]> trainingOutputs, List<float> trainingSetWeighting, int N_threads)
    {
        if (net.trainingInProgress == false)
        {
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            if (batch_size > no_training_sets)
            {
                if (!suppressMessages) print("Warning: batch size (" + batch_size + ") greater than the # of training sets (" + no_training_sets + "). Reducing batch size to " + no_training_sets);
                batch_size = no_training_sets;
            }
            CleanupTrainingNativeArrays();
            NativeArray<Noedify.ActivationFunction> activationFunctionList = new NativeArray<Noedify.ActivationFunction>(net.LayerCount(), Allocator.Persistent);
            NativeArray<Noedify.PoolingType> poolingTypeList = new NativeArray<Noedify.PoolingType>(net.LayerCount(), Allocator.Persistent);
            NativeArray<Noedify.LayerType> layerTypeList = new NativeArray<Noedify.LayerType>(net.LayerCount(), Allocator.Persistent);
            nativeArrayCleanupList.Add(activationFunctionList);
            nativeArrayCleanupList.Add(poolingTypeList);
            nativeArrayCleanupList.Add(layerTypeList);
            NativeArray<int> layerSizeList = NativeAllocInt(net.LayerCount());
            NativeArray<float> trainingInputs_batchSet = NativeAllocFloat(batch_size * net.layers[0].layerSize);
            NativeArray<float> trainingOutputs_batchSet = NativeAllocFloat(batch_size * net.layers[net.LayerCount() - 1].layerSize);
            NativeArray<float> batch_cost = NativeAllocFloat(batch_size);
            NativeArray<int> CNN_nodes_per_filter_par = NativeAllocInt(net.LayerCount() - 1);
            NativeArray<int> CNN_weights_per_filter_par = NativeAllocInt(net.LayerCount() - 1);
            NativeArray<int> CNN_no_filters_par = NativeAllocInt(net.LayerCount() - 1);
            net.networkWeights_gradients_par = NativeAllocFloat(net.total_no_weights * batch_size);
            net.networkBiases_gradients_par = NativeAllocFloat(net.total_no_biases * batch_size);
            NativeArray<float> nodeOutputs_par = NativeAllocFloat(batch_size * net.total_no_nodes);
            NativeArray<float> nodeInputs_par = NativeAllocFloat(batch_size * net.total_no_nodes);
            NativeArray<float> deltas_par = NativeAllocFloat(batch_size * net.total_no_activeNodes);
            NativeArray<int> CNN_no_conns_per_node = NativeAllocInt(net.LayerCount() - 1);

            InitializeNetTempNativeArrays(
                net,
                ref activationFunctionList,
                ref layerTypeList,
                ref layerSizeList,
                ref CNN_nodes_per_filter_par,
                ref CNN_weights_per_filter_par,
                ref CNN_no_filters_par,
                ref CNN_no_conns_per_node,
                ref poolingTypeList);

            List<float[]> trainingInputs_flat = Noedify_Utils.FlattenDataset(trainingInputs);

#if TIME_PROFILE
            sw.Stop();
            print("----setup time: " + sw.ElapsedMilliseconds);
            sw.Reset();
            sw.Start();
#endif


            for (int epoch = 0; epoch < no_epochs; epoch++)
            {
#if TIME_PROFILE
                System.Diagnostics.Stopwatch sw_batch = new System.Diagnostics.Stopwatch();
                sw_batch.Start();
#endif
                int[] trainingSets = new int[no_training_sets];
                for (int i = 0; i < no_training_sets; i++)
                    trainingSets[i] = i;
                trainingSets = Noedify_Utils.Shuffle(trainingSets);

#if FIX_0_TRAININGSET
                trainingSets[0] = 0;
#endif
#if FIX_1_TRAININGSET
                trainingSets[1] = 6;
#endif
                for (int batch = 0; batch < batch_size; batch++)
                {
                    for (int i = 0; i < net.layers[0].layerSize; i++)
                    {
                        trainingInputs_batchSet[batch * net.layers[0].layerSize + i] = trainingInputs_flat[trainingSets[batch]][i];
                    }
                    for (int i = 0; i < net.layers[net.LayerCount() - 1].layerSize; i++)
                        trainingOutputs_batchSet[batch * net.layers[net.LayerCount() - 1].layerSize + i] = trainingOutputs[trainingSets[batch]][i];
                }

                for (int batch = 0; batch < batch_size; batch++)
                    batch_cost[batch] = 0;

                for (int i = 0; i < net.total_no_weights * batch_size; i++)
                    net.networkWeights_gradients_par[i] = 0;
                for (int i = 0; i < net.total_no_biases * batch_size; i++)
                    net.networkBiases_gradients_par[i] = 0;

                TrainNetwork_Job trainNetwork_Job = new TrainNetwork_Job();

                trainNetwork_Job.activationFunction = activationFunctionList;
                trainNetwork_Job.costFunction = costFunction;
                trainNetwork_Job.layerTypes = layerTypeList;
                trainNetwork_Job.layerPooling = poolingTypeList;
                trainNetwork_Job.layerSize = layerSizeList;
                trainNetwork_Job.networkWeights = net.networkWeights_par;
                trainNetwork_Job.networkBiases = net.networkBiases_par;
                trainNetwork_Job.weightIndeces_start = net.weightIdx_start;
                trainNetwork_Job.biasIndeces_start = net.biasIdx_start;
                trainNetwork_Job.activeNodeIndeces_start = net.activeNodeIdx_start;
                trainNetwork_Job.nodeIndeces_start = net.nodeIdx_start;
                trainNetwork_Job.nodeInputs = nodeInputs_par;
                trainNetwork_Job.nodeOutputs = nodeOutputs_par;
                trainNetwork_Job.deltas = deltas_par;
                trainNetwork_Job.networkWeightGradients = net.networkWeights_gradients_par;
                trainNetwork_Job.networkBiasGradients = net.networkBiases_gradients_par;
                trainNetwork_Job.connectionMask = net.connectionMask_par;
                trainNetwork_Job.connectionMaskStart_index = net.connectionMaskIndeces_start;
                trainNetwork_Job.CNN_nodes_per_filter = CNN_nodes_per_filter_par;
                trainNetwork_Job.CNN_weights_per_filter = CNN_weights_per_filter_par;
                trainNetwork_Job.CNN_no_filters = CNN_no_filters_par;
                trainNetwork_Job.CNN_conns_per_node = CNN_no_conns_per_node;
                trainNetwork_Job.trainingInputs = trainingInputs_batchSet;
                trainNetwork_Job.trainingOutputs = trainingOutputs_batchSet;
                trainNetwork_Job.total_no_active_nodes = net.total_no_activeNodes;
                trainNetwork_Job.total_no_nodes = net.total_no_nodes;
                trainNetwork_Job.network_input_size = net.layers[0].layerSize;
                trainNetwork_Job.batch_cost = batch_cost;
                trainNetwork_Job.debug_job = debug.ConvertForPar();
                trainNetwork_Job.fineTuningLayerLimit_job = fineTuningLayerLimit;

                JobHandle training_handle;
#if TIME_PROFILE
                sw_batch.Stop();
                print("----batch setup time: " + sw_batch.ElapsedMilliseconds);
                sw_batch.Reset();
                sw_batch.Start();
#endif

                int batched_per_thread = Mathf.CeilToInt(batch_size / N_threads);
                training_handle = trainNetwork_Job.Schedule(batch_size, batched_per_thread);
                activeJob = training_handle;

                net.trainingInProgress = true;

                int timeoutCounter = 0;
                while (true)
                {
                    yield return null;
                    timeoutCounter++;
                    if (training_handle.IsCompleted)
                    {
                        break;
                    }
                    else if (timeoutCounter > 100000)
                    {
                        if (!suppressMessages) print("Error: Training timeout reached!");
                        goto Error;
                    }
                }
                training_handle.Complete();
#if TIME_PROFILE
                sw_batch.Stop();
                print("----batch train time: " + sw_batch.ElapsedMilliseconds);
                sw_batch.Reset();
                sw_batch.Start();
#endif

                float epoch_cost = 0;

                for (int batch = 0; batch < batch_size; batch++)
                    epoch_cost += batch_cost[batch];
                epoch_cost /= batch_size;
                if (epoch % 10 == 0 | epoch == no_epochs - 1)
                {
                    if (!suppressMessages) print("epoch " + (epoch + 1) + "/" + no_epochs + ", cost: " + epoch_cost);
                }
                cost_report[epoch] = epoch_cost;

                float[] weighting = new float[batch_size];
                for (int batch = 0; batch < batch_size; batch++)
                    weighting[batch] = trainingSetWeighting[trainingSets[batch]];
                Noedify.NN_LayerGradients[] netGradients = AccumulateGradients(net, trainingRate, net.networkWeights_gradients_par, net.networkBiases_gradients_par, batch_size, weighting);

                if (debug.print_bias_Gradients && ((epoch % 10 == 0)) | (epoch == (no_epochs - 1)))
                {
                    for (int l = 1; l < net.LayerCount(); l++)
                    {
                        string biasGradientString = "Epoch " + epoch + " bias gradients layer " + l + ": ";
                        if (layerTypeList[l] == Noedify.LayerType.Convolutional2D)
                        {
                            for (int f = 0; f < CNN_no_filters_par[l - 1]; f++)
                                biasGradientString += "(b" + f + "=" + net.networkBiases_par[net.biasIdx_start[l - 1] + f] + ")" + netGradients[l - 1].bias_gradients.valuesConv[f] + " ";
                        }
                        else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D) { }
                        else
                        {
                            for (int j = 0; j < net.layers[l].layerSize; j++)
                                biasGradientString += "(b" + j + "=" + net.networkBiases_par[net.biasIdx_start[l - 1] + j] + ")" + netGradients[l - 1].bias_gradients.values[j] + " ";
                        }
                        print(biasGradientString);
                    }
                }
                if (debug.print_weight_Gradients && ((epoch % 10 == 0) | (epoch == no_epochs - 1)))
                {
                    for (int l = 1; l < net.LayerCount(); l++)
                    {
                        if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
                        {
                            for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                            {
                                string weightGradientsDebug = "Epoch " + epoch + " layer " + l + " from filter " + f + ": ";
                                for (int j = 0; j < net.layers[l].conv2DLayer.N_weights_per_filter; j++)
                                {
                                    weightGradientsDebug += "(w_" + f + "_" + j + "=" + net.networkWeights_par[net.weightIdx_start[l - 1] + f * net.layers[l].conv2DLayer.N_weights_per_filter + j] + ")" + netGradients[l - 1].weight_gradients.valuesConv[f, j] + " ";
                                }
                                print(weightGradientsDebug);
                            }
                        }
                        else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D) { }
                        else
                        {
                            for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                            {
                                if (i > 10)
                                    break;
                                string weightGradientsDebug = "Epoch " + epoch + " layer " + l + " from node " + i + ": ";
                                for (int j = 0; j < net.layers[l].layerSize; j++)
                                {
                                    weightGradientsDebug += "(w_" + i + "_" + j + "=" + net.networkWeights_par[net.weightIdx_start[l - 1] + i * net.layers[l].layerSize + j] + ")" + netGradients[l - 1].weight_gradients.values[i, j] + " ";
                                }
                                print(weightGradientsDebug);
                            }
                        }
                    }
                }
                net.ApplyGradients(netGradients, batch_size, true, fineTuningLayerLimit);
#if TIME_PROFILE
                sw_batch.Stop();
                print("----batch gradient application time: " + sw_batch.ElapsedMilliseconds);
                sw_batch.Reset();
                sw_batch.Start();
#endif
                if (epoch_cost < costThreshold)
                {
                    if (!suppressMessages) print("Cost Threshold reached (" + costThreshold + "). Training Complete.");
                    break;
                }
            }

            Error:
            net.OffloadNativeParameterArrays();
            CleanupTrainingNativeArrays();
            net.Cleanup_Par();
            net.trainingInProgress = false;
            trainingInProgress = false;

        }
        else
        {
            if (!suppressMessages) print("Error: Training initiated before previous job competed");
            trainingInProgress = false;
            net.Cleanup_Par();
        }
        yield return null;
    }
#endif
#if NOEDIFY_BURST
    [BurstCompile(CompileSynchronously = true)]
#endif
#if NOEDIFY_NOTRAIN
    struct TrainNetwork_Job : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<Noedify.ActivationFunction> activationFunction;
        public CostFunction costFunction;
        [NativeDisableParallelForRestriction] public NativeArray<Noedify.LayerType> layerTypes;
        [NativeDisableParallelForRestriction] public NativeArray<Noedify.PoolingType> layerPooling;
        [NativeDisableParallelForRestriction] public NativeArray<int> layerSize;

        [NativeDisableParallelForRestriction] public NativeArray<float> networkWeights;
        [NativeDisableParallelForRestriction] public NativeArray<float> networkBiases;
        [NativeDisableParallelForRestriction] public NativeArray<int> weightIndeces_start;
        [NativeDisableParallelForRestriction] public NativeArray<int> biasIndeces_start;
        [NativeDisableParallelForRestriction] public NativeArray<int> activeNodeIndeces_start;
        [NativeDisableParallelForRestriction] public NativeArray<int> nodeIndeces_start;

        [NativeDisableParallelForRestriction] public NativeArray<float> nodeInputs;
        [NativeDisableParallelForRestriction] public NativeArray<float> nodeOutputs;
        [NativeDisableParallelForRestriction] public NativeArray<float> deltas;

        [NativeDisableParallelForRestriction] public NativeArray<float> networkWeightGradients;
        [NativeDisableParallelForRestriction] public NativeArray<float> networkBiasGradients;

        [NativeDisableParallelForRestriction] public NativeArray<int> connectionMask;
        [NativeDisableParallelForRestriction] public NativeArray<int> connectionMaskStart_index;

        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_nodes_per_filter;
        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_weights_per_filter;
        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_no_filters;
        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_conns_per_node;

        [NativeDisableParallelForRestriction] public NativeArray<float> trainingInputs;
        [NativeDisableParallelForRestriction] public NativeArray<float> trainingOutputs;

        [NativeDisableParallelForRestriction] public NativeArray<float> batch_cost;

        public DebugReport_Par debug_job;
        public int total_no_active_nodes;
        public int total_no_nodes;
        public int network_input_size;
        public int fineTuningLayerLimit_job;

        public void Execute(int batch)
        {
            // Forward Evaluate
            for (int j = 0; j < network_input_size; j++)
                nodeOutputs[batch * total_no_nodes + j] = trainingInputs[batch * network_input_size + j];
            for (int l = 1; l < layerSize.Length; l++)
            {
                EvaluateLayer_Job(
                l,
                layerTypes[l],
                activationFunction[l],
                layerPooling[l],
                layerSize,
                ref nodeInputs,
                ref nodeOutputs,
                networkWeights,
                networkBiases,
                connectionMask,
                nodeIndeces_start,
                weightIndeces_start,
                biasIndeces_start,
                connectionMaskStart_index,
                CNN_nodes_per_filter,
                CNN_weights_per_filter,
                CNN_conns_per_node,
                batch * total_no_nodes);
            }
#if !NOEDIFY_BURST
            if (debug_job.print_nodeOutputs)
                for (int l = 0; l < layerSize.Length; l++)
                    Noedify.PrintArrayLine("(Job " + batch + ") layer " + l + " node outputs", nodeOutputs, new int[2] { batch * total_no_nodes + nodeIndeces_start[l], batch * total_no_nodes + nodeIndeces_start[l] + layerSize[l] }, true);
            if (debug_job.print_nodeInputs)
                for (int l = 0; l < layerSize.Length; l++)
                    Noedify.PrintArrayLine("(Job " + batch + ") layer " + l + " node inputs", nodeInputs, new int[2] { batch * total_no_nodes + nodeIndeces_start[l], batch * total_no_nodes + nodeIndeces_start[l] + layerSize[l] }, true);
#endif
            // Calculate cost/deltas
            int output_layer_no = layerSize.Length - 1;
            int output_layer_size = layerSize[output_layer_no];

            DeltaCost(ref deltas, batch * total_no_active_nodes + activeNodeIndeces_start[output_layer_no - 1], nodeOutputs, batch * total_no_nodes + nodeIndeces_start[output_layer_no], trainingOutputs, batch * output_layer_size, output_layer_size, costFunction);
            batch_cost[batch] = Cost(nodeOutputs, batch * total_no_nodes + nodeIndeces_start[layerSize.Length - 1], trainingOutputs, batch * output_layer_size, output_layer_size, costFunction);
#if !NOEDIFY_BURST
            if (debug_job.print_outputError)
            {
                string errorCalcString = "(Job " + batch + ") Cost Calculation: ";
                for (int j = 0; j < layerSize[output_layer_no]; j++)
                {
                    float nodeOut = nodeOutputs[batch * total_no_nodes + nodeIndeces_start[output_layer_no] + j];
                    float trainOut = trainingOutputs[batch * output_layer_size + j];
                    float delTest = deltas[batch * total_no_active_nodes + activeNodeIndeces_start[output_layer_no - 1] + j];
                    errorCalcString += "(" + j + ")[" + nodeOut + " - " + trainOut + " = " + delTest + "] , ";
                }
                print(errorCalcString);
            }
#endif
            NativeArray<float> delta_z = new NativeArray<float>(layerSize[output_layer_no], Allocator.Temp);
            ApplyDeltaActivationFunction(ref delta_z, 0, nodeInputs, batch * total_no_nodes + nodeIndeces_start[output_layer_no], output_layer_size, activationFunction[output_layer_no]);
            for (int i = 0; i < layerSize[output_layer_no]; i++)
                deltas[batch * total_no_active_nodes + activeNodeIndeces_start[output_layer_no - 1] + i] *= delta_z[i];
#if !NOEDIFY_BURST
            if (debug_job.print_deltas)
            {
                string outStringDelta = "Output layer deltas: ";
                for (int i = 0; i < layerSize[output_layer_no]; i++)
                    outStringDelta += "(" + delta_z[i] + ")" + deltas[batch * total_no_active_nodes + activeNodeIndeces_start[output_layer_no - 1] + i] + ", ";
                print(outStringDelta);
            }
#endif
            //Calculate output layer gradients
            CalculateGradientsLayer_Job(
                output_layer_no,
                batch,
                total_no_active_nodes,
                total_no_nodes,
                networkWeights.Length,
                networkBiases.Length,
                layerTypes[output_layer_no],
                layerSize,
                nodeOutputs,
                ref networkWeightGradients,
                ref networkBiasGradients,
                connectionMask,
                activeNodeIndeces_start,
                nodeIndeces_start,
                weightIndeces_start,
                biasIndeces_start,
                connectionMaskStart_index,
                CNN_nodes_per_filter,
                CNN_weights_per_filter,
                CNN_conns_per_node,
                deltas);

            // Backprogate/Calculate gradients for hidden layers
            for (int l = (layerSize.Length - 2); l > 0; l--)
            {
                int thisLayer = l;
                int nextLayer = l + 1;
                int prevLayer = l - 1;
                // Backpropagate

                BackPropagateLayer_Job(
                    thisLayer,
                    nextLayer,
                    output_layer_no,
                    batch,
                    total_no_active_nodes,
                    total_no_nodes,
                    activationFunction[thisLayer],
                    layerPooling[nextLayer],
                    layerTypes,
                    layerSize,
                    connectionMask,
                    activeNodeIndeces_start,
                    nodeIndeces_start,
                    nodeInputs,
                    nodeOutputs,
                    networkWeights,
                    weightIndeces_start,
                    connectionMaskStart_index,
                    CNN_nodes_per_filter,
                    CNN_weights_per_filter,
                    CNN_conns_per_node,
                    ref deltas);
#if !NOEDIFY_BURST
                if (debug_job.print_deltas)
                {
                    string outStringDelta = "Layer " + l + " deltas: ";
                    for (int i = 0; i < layerSize[thisLayer]; i++)
                        outStringDelta += deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + i] + ", ";
                    print(outStringDelta);
                }
#endif
                // Calculate gradients
                CalculateGradientsLayer_Job(
                    thisLayer,
                    batch,
                    total_no_active_nodes,
                    total_no_nodes,
                    networkWeights.Length,
                    networkBiases.Length,
                    layerTypes[thisLayer],
                    layerSize,
                    nodeOutputs,
                    ref networkWeightGradients,
                    ref networkBiasGradients,
                    connectionMask,
                    activeNodeIndeces_start,
                    nodeIndeces_start,
                    weightIndeces_start,
                    biasIndeces_start,
                    connectionMaskStart_index,
                    CNN_nodes_per_filter,
                    CNN_weights_per_filter,
                    CNN_conns_per_node,
                    deltas);

                if (l == fineTuningLayerLimit_job)
                    break;

            }
        }
    }
#endif
#if NOEDIFY_BURST
    [BurstCompile(CompileSynchronously = true)]
#endif
#if NOEDIFY_NORELEASE
    struct EvaluateNetwork_Job : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<Noedify.ActivationFunction> activationFunction;
        [NativeDisableParallelForRestriction] public NativeArray<Noedify.LayerType> layerTypes;
        [NativeDisableParallelForRestriction] public NativeArray<Noedify.PoolingType> layerPooling;
        [NativeDisableParallelForRestriction] public NativeArray<int> layerSize;

        [NativeDisableParallelForRestriction] public NativeArray<float> networkWeights;
        [NativeDisableParallelForRestriction] public NativeArray<float> networkBiases;
        [NativeDisableParallelForRestriction] public NativeArray<int> weightIndeces_start;
        [NativeDisableParallelForRestriction] public NativeArray<int> biasIndeces_start;
        [NativeDisableParallelForRestriction] public NativeArray<int> activeNodeIndeces_start;
        [NativeDisableParallelForRestriction] public NativeArray<int> nodeIndeces_start;

        [NativeDisableParallelForRestriction] public NativeArray<float> nodeInputs;
        [NativeDisableParallelForRestriction] public NativeArray<float> nodeOutputs;

        [NativeDisableParallelForRestriction] public NativeArray<int> connectionMask;
        [NativeDisableParallelForRestriction] public NativeArray<int> connectionMaskStart_index;

        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_nodes_per_filter;
        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_weights_per_filter;
        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_no_filters;
        [NativeDisableParallelForRestriction] public NativeArray<int> CNN_conns_per_node;

        [NativeDisableParallelForRestriction] public NativeArray<float> networkInputs;
        [NativeDisableParallelForRestriction] public NativeArray<float> networkOutputs;

        public int total_no_active_nodes;
        public int network_input_size;

        public void Execute(int batch)
        {
            int output_layer_no = layerSize.Length - 1;

            // Forward Evaluate
            for (int j = 0; j < network_input_size; j++)
                nodeOutputs[j] = networkInputs[batch * network_input_size + j];


            for (int l = 1; l < layerSize.Length; l++)
            {
                EvaluateLayer_Job(
                l,
                layerTypes[l],
                activationFunction[l],
                layerPooling[l],
                layerSize,
                ref nodeInputs,
                ref nodeOutputs,
                networkWeights,
                networkBiases,
                connectionMask,
                nodeIndeces_start,
                weightIndeces_start,
                biasIndeces_start,
                connectionMaskStart_index,
                CNN_nodes_per_filter,
                CNN_weights_per_filter,
                CNN_conns_per_node);
            }

            for (int j = 0; j < layerSize[output_layer_no]; j++)
                networkOutputs[j] = nodeOutputs[nodeIndeces_start[output_layer_no] + j];
        }
    }
#endif
    void Evaluate_CBuff(Noedify.Net net, Noedify.InputArray inputs, ComputeShader _shader)
    {
        Debug.Log("Evaluating with GPU");
        if (net.trainingInProgress == false)
        {
            // System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            System.Diagnostics.Stopwatch sw1 = new System.Diagnostics.Stopwatch();
            sw1.Start();
            CleanupTrainingCBuffers();

            ComputeBuffer nodeOutputs_cbuf = new ComputeBuffer(net.total_no_nodes, sizeof(float));
            ComputeBuffer nodeInputs_cbuf = new ComputeBuffer(net.total_no_nodes, sizeof(float));

            cBufferCleanupList.Add(nodeOutputs_cbuf);
            cBufferCleanupList.Add(nodeInputs_cbuf);

            int threads;
            int kernel_MatMul = _shader.FindKernel("MatMul");
            int kernel_AF = _shader.FindKernel("ApplyAF");
            int kernel_ApplyBias = _shader.FindKernel("ApplyBias");
            int kernel_I2_Conv2D = _shader.FindKernel("I2D_Conv2D");
            int kernel_I2_Conv3D = _shader.FindKernel("I2D_Conv3D");
            int kernel_I2_TranspConv2D = _shader.FindKernel("I2D_TranspConv2D");
            int kernel_I2_TranspConv3D = _shader.FindKernel("I2D_TranspConv3D");
            //int kernel_I2D_BatchNorm2D_Mean = _shader.FindKernel("I2D_BatchNorm2D_Mean");
            //int kernel_I2D_BatchNorm2D_Std1 = _shader.FindKernel("I2D_BatchNorm2D_Std1");
            //int kernel_I2D_BatchNorm2D_Std2 = _shader.FindKernel("I2D_BatchNorm2D_Std2");
            //int kernel_I2D_BatchNorm2D = _shader.FindKernel("I2D_BatchNorm2D");
            //int kernel_I2D_BatchNorm3D = _shader.FindKernel("I2D_BatchNorm3D");


            float[] inputsFlat = inputs.FlattenArray();
            if (inputsFlat.Length == net.nodeIdx_start[1])
            {
                float[] netOutputsInit = new float[net.total_no_nodes];
                for (int j = 0; j < inputsFlat.Length; j++)
                    netOutputsInit[j] = inputsFlat[j];

                nodeOutputs_cbuf.SetData(netOutputsInit);
                _shader.SetBuffer(kernel_MatMul, "nodeOutputs", nodeOutputs_cbuf);
            }
            else
            {
                Debug.LogError("(Noedify_Solver.Evaluate_CBuff): Node input size (" + inputsFlat.Length + ") incompatable with input layer size (" + net.nodeIdx_start[1] + ")");
                goto Error;
            }
            // sw.Start();

            int lastEvalLyr = net.LayerCount();
            if (lastLayer >= 0)
                lastEvalLyr = lastLayer + 1;
            sw1.Stop();
            Debug.Log("***Evaluation Cbuf initialization: " + sw1.ElapsedMilliseconds + " ms");

            //print("evaluating up to (including) layer " + (lastEvalLyr - 1));
            for (int l = 1; l < lastEvalLyr; l++)
            {
                UnityEngine.Profiling.Profiler.BeginSample("layer " + l + " (" + net.layers[l].name + ")");

                sw1.Reset(); sw1.Start();
                print("forward evaluation layer " + l + " (" + net.layers[l].name + ")");
                Noedify.LayerType layerType = net.layers[l].layer_type;
                int layerSize = net.layers[l].layerSize;
                _shader.SetInt("current_layer", l);
                _shader.SetInt("layerSize", net.layers[l].layerSize);
                _shader.SetInt("node_start_index", net.nodeIdx_start[l]);

                _shader.SetInt("activeNodeIdx", net.activeNodeIdx_start[l - 1]);
                if (l > 1)
                    _shader.SetInt("activeNodeIdx_prevlyr", net.activeNodeIdx_start[l - 2]);
                _shader.SetInt("nodeIdx", net.nodeIdx_start[l]);
                _shader.SetInt("nodeIdx_prevlyr", net.nodeIdx_start[l - 1]);

                if (Noedify_Utils.Is3DLayerType(net.layers[l]))
                    _shader.SetInt("channelSize", net.layers[l].layerSize3D[0] * net.layers[l].layerSize3D[1] * net.layers[l].layerSize3D[2]);
                else
                    _shader.SetInt("channelSize", net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1]);
                _shader.SetInt("weightsIdx", net.weightIdx_start[l - 1]);
                _shader.SetInt("biasIdx", net.biasIdx_start[l - 1]);
                _shader.SetInt("connectionsIdx", net.connectionsIdx_start[l - 1]);
                _shader.SetFloat("eps", net.layers[l].bn_eps);

                switch (layerType)
                {
                    case (Noedify.LayerType.Convolutional2D):
                        {
                            _shader.SetInt("apply_bias_layerSize", net.layers[l].layerSize);
                            _shader.SetInt("apply_bias_param_offset", net.activeNodeIdx_start[l - 1]);
                            _shader.SetInt("apply_bias_node_offset", net.nodeIdx_start[l]);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_out", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_param", net.biasMask_cbuf);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)8);
                            _shader.Dispatch(kernel_ApplyBias, threads, 1, 1);

                            _shader.SetInt("connPerNode", net.layers[l].conv2DLayer.N_connections_per_node);
                            _shader.SetInt("filterSize", net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1]);
                            _shader.SetInt("no_filters_prevlyr", net.layers[l - 1].conv2DLayer.no_filters);


                            _shader.SetBuffer(kernel_I2_Conv2D, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv2D, "nodeTrack", net.nodeTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv2D, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv2D, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv2D, "weights", net.networkWeights_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv2D, "connections", net.connections_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv2D, "connectionsInFilter", net.connectionsInFilter_cbuf);

                            // ComputeBuffer debugBuff = new ComputeBuffer(1000, sizeof(float));
                            // _shader.SetBuffer(kernel_I2_Conv2D, "debugFloats", debugBuff);

                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2_Conv2D, threads, 1, 1);

                            /*
                            float[] debugFloats = new float[100];
                            debugBuff.GetData(debugFloats);
                            string stringOutn = "GPU cif node 0: ";
                            for (int i = 0; i < 100; i++)
                            {
                                int prevFilterno = Mathf.FloorToInt(i / (net.layers[l - 1].layerSize2D[0] * net.layers[l - 1].layerSize2D[1]));
                                stringOutn += "(" + ((float)i / (float)net.layers[l].conv2DLayer.N_connections_per_node) + ")" + debugFloats[i] + ",";
                            }
                            */
                            //Debug.Log(stringOutn);
                            //print("processing Conv2D layer " + l + " with " + threads + " threads");
                            break;
                        }
                    case (Noedify.LayerType.Convolutional3D):
                        {

                            _shader.SetInt("apply_bias_layerSize", net.layers[l].layerSize);
                            _shader.SetInt("apply_bias_param_offset", net.activeNodeIdx_start[l - 1]);
                            _shader.SetInt("apply_bias_node_offset", net.nodeIdx_start[l]);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_out", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_param", net.biasMask_cbuf);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)8);
                            _shader.Dispatch(kernel_ApplyBias, threads, 1, 1);

                            _shader.SetInt("connPerNode", net.layers[l].conv3DLayer.N_connections_per_node);
                            _shader.SetInt("filterSize", net.layers[l].conv3DLayer.filterSize[0] * net.layers[l].conv3DLayer.filterSize[1] * net.layers[l].conv3DLayer.filterSize[2]);
                            _shader.SetInt("no_filters_prevlyr", net.layers[l - 1].conv3DLayer.no_filters);


                            _shader.SetBuffer(kernel_I2_Conv3D, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv3D, "nodeTrack", net.nodeTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv3D, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv3D, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv3D, "weights", net.networkWeights_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv3D, "connections", net.connections_cbuf);
                            _shader.SetBuffer(kernel_I2_Conv3D, "connectionsInFilter", net.connectionsInFilter_cbuf);

                            // ComputeBuffer debugBuff = new ComputeBuffer(1000, sizeof(float));
                            // _shader.SetBuffer(kernel_I2_Conv3D, "debugFloats", debugBuff);

                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2_Conv3D, threads, 1, 1);

                            /*
                            float[] debugFloats = new float[100];
                            debugBuff.GetData(debugFloats);
                            string stringOutn = "GPU cif node 0: ";
                            for (int i = 0; i < 100; i++)
                            {
                                int prevFilterno = Mathf.FloorToInt(i / (net.layers[l - 1].layerSize2D[0] * net.layers[l - 1].layerSize2D[1]));
                                stringOutn += "(" + ((float)i / (float)net.layers[l].conv2DLayer.N_connections_per_node) + ")" + debugFloats[i] + ",";
                            }
                            */
                            //Debug.Log(stringOutn);
                            //print("processing Conv2D layer " + l + " with " + threads + " threads");
                            break;
                        }
                    case (Noedify.LayerType.Pool2D):
                        {
                            /*
                              for (int j = 0; j < layerSize[layer_no]; j++)
                    {
                        int connsMade = 0;
                        if (poolingType == Noedify.PoolingType.Max)
                            nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = -100;
                        else if (poolingType == Noedify.PoolingType.Avg)
                            nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = 0;

                        for (int i = 0; i < layerSize[layer_no - 1]; i++)
                        {
                            int connectionMask_j = connectionMask[connectionMaskStart_index[layer_no - 1] + i * layerSize[layer_no] + j];
                            if (connectionMask_j >= 0)
                            {
                                connsMade++;
                                if (poolingType == Noedify.PoolingType.Avg)
                                    nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] += nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i];
                                else if (poolingType == Noedify.PoolingType.Max)
                                {
                                    if (nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i] > nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j])
                                    {
                                        nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i];
                                    }
                                }
                            }
                            if (connsMade >= CNN_conns_per_node[layer_no - 1])
                                break;
                        }
                        if (poolingType == Noedify.PoolingType.Avg)
                            nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] /= CNN_weights_per_filter[layer_no - 1];
                    }
                            */
                            break;
                        }
                    case (Noedify.LayerType.TranspConvolutional2D):
                        {

                            _shader.SetInt("apply_bias_layerSize", net.layers[l].layerSize);
                            _shader.SetInt("apply_bias_param_offset", net.activeNodeIdx_start[l - 1]);
                            _shader.SetInt("apply_bias_node_offset", net.nodeIdx_start[l]);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_out", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_param", net.biasMask_cbuf);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)8);
                            _shader.Dispatch(kernel_ApplyBias, threads, 1, 1);

                            _shader.SetInt("connPerNode", net.layers[l].conv2DLayer.N_connections_per_node);
                            _shader.SetInt("weightsPerFilter", net.layers[l].conv2DLayer.N_weights_per_filter);
                            _shader.SetInt("filterSize", net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1]);
                            _shader.SetInt("kernelSize_prevlyr", net.layers[l - 1].layerSize2D[0] * net.layers[l - 1].layerSize2D[1]);
                            _shader.SetInt("layerSize_prevlyr", net.layers[l - 1].layerSize);
                            _shader.SetInt("no_filters_prevlyr", net.layers[l - 1].conv2DLayer.no_filters);

                            //ComputeBuffer debugBuff = new ComputeBuffer(100, sizeof(float));
                            //_shader.SetBuffer(kernel_I2_TranspConv2D, "debugFloats", debugBuff);

                            _shader.SetBuffer(kernel_I2_TranspConv2D, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv2D, "nodeTrack", net.nodeTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv2D, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv2D, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv2D, "weights", net.networkWeights_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv2D, "connections", net.connections_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv2D, "connectionsInFilter", net.connectionsInFilter_cbuf);

                            threads = Mathf.CeilToInt((float)(net.layers[l].layerSize) / (float)16);
                            _shader.Dispatch(kernel_I2_TranspConv2D, threads, 1, 1);

                            break;
                        }
                    case (Noedify.LayerType.TranspConvolutional3D):
                        {
                            _shader.SetInt("apply_bias_layerSize", net.layers[l].layerSize);
                            _shader.SetInt("apply_bias_param_offset", net.activeNodeIdx_start[l - 1]);
                            _shader.SetInt("apply_bias_node_offset", net.nodeIdx_start[l]);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_out", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_param", net.biasMask_cbuf);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)8);
                            _shader.Dispatch(kernel_ApplyBias, threads, 1, 1);

                            _shader.SetInt("connPerNode", net.layers[l].conv3DLayer.N_connections_per_node);
                            _shader.SetInt("weightsPerFilter", net.layers[l].conv3DLayer.N_weights_per_filter);
                            _shader.SetInt("filterSize", net.layers[l].conv3DLayer.filterSize[0] * net.layers[l].conv3DLayer.filterSize[1] * net.layers[l].conv3DLayer.filterSize[2]);
                            _shader.SetInt("kernelSize_prevlyr", net.layers[l - 1].layerSize3D[0] * net.layers[l - 1].layerSize3D[1] * net.layers[l - 1].layerSize3D[2]);
                            _shader.SetInt("layerSize_prevlyr", net.layers[l - 1].layerSize);
                            _shader.SetInt("no_filters_prevlyr", net.layers[l - 1].conv3DLayer.no_filters);

                            //ComputeBuffer debugBuff = new ComputeBuffer(100, sizeof(float));
                            //_shader.SetBuffer(kernel_I2_TranspConv3D, "debugFloats", debugBuff);

                            _shader.SetBuffer(kernel_I2_TranspConv3D, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv3D, "nodeTrack", net.nodeTrack_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv3D, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv3D, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv3D, "weights", net.networkWeights_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv3D, "connections", net.connections_cbuf);
                            _shader.SetBuffer(kernel_I2_TranspConv3D, "connectionsInFilter", net.connectionsInFilter_cbuf);

                            threads = Mathf.CeilToInt((float)(net.layers[l].layerSize) / (float)16);
                            _shader.Dispatch(kernel_I2_TranspConv3D, threads, 1, 1);

                            break;
                        }
                    case (Noedify.LayerType.BatchNorm2D):
                        {

                            float[] filterMean = new float[net.layers[l].in_channels];
                            float[] filterVar = new float[net.layers[l].in_channels];

                            float[] batchNorm_outputs_prevlyr = new float[net.layers[l - 1].layerSize];
                            nodeOutputs_cbuf.GetData(batchNorm_outputs_prevlyr, 0, net.nodeIdx_start[l - 1], net.layers[l - 1].layerSize);
                            float[] batchNorm_inputs = new float[net.layers[l].layerSize];

                            // Noedify_Utils.PrintArray(batchNorm_inputs, "BatchNorm inputs");

                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                //print("filterMean[" + layer.conv2DLayer.filterTrack[j] + "](j=" + j + ") += " + inputs[j] + " = " + (filterMean[layer.conv2DLayer.filterTrack[j]] + inputs[j]));
                                filterMean[net.layers[l].conv2DLayer.filterTrack[j]] += batchNorm_outputs_prevlyr[j];
                            }

                            //print("filtermean: " + filterMean[0] + " / " + (layer.layerSize2D[0] * layer.layerSize2D[1]) + " + " + filterMean[0]);

                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                filterMean[c] /= (net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1]);
                            /*
                            string outString = "filterMean: ";
                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                outString += filterMean[c] + ", ";
                            print(outString);
                            */
                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                int filter_no = net.layers[l].conv2DLayer.filterTrack[j];
                                float varTemp = (batchNorm_outputs_prevlyr[j] - filterMean[filter_no]);
                                filterVar[filter_no] += varTemp * varTemp;
                            }
                            /*
                            string outStringVar = "filterVar: ";
                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                outStringVar += filterVar[c] + ", ";
                            print(outStringVar);
                            */
                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                filterVar[c] = Mathf.Sqrt(filterVar[c] / ((net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1])) + net.layers[l].bn_eps);

                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                int filter_no = net.layers[l].conv2DLayer.filterTrack[j];
                                float inputHat = (batchNorm_outputs_prevlyr[j] - filterMean[filter_no]) / filterVar[filter_no];
                                batchNorm_inputs[j] = net.layers[l].weights.values[0, filter_no] * inputHat + net.layers[l].biases.values[filter_no];
                            }

                            nodeInputs_cbuf.SetData(batchNorm_inputs, 0, net.nodeIdx_start[l], net.layers[l].layerSize);

#region BathNormGPU
                            /*
                            ComputeBuffer batchNorm2D_filterMean = new ComputeBuffer(net.layers[l].in_channels, sizeof(float));
                            ComputeBuffer batchNorm2D_filterStd = new ComputeBuffer(net.layers[l].in_channels, sizeof(float));

                            ComputeBuffer debug_int_cb = new ComputeBuffer(1, sizeof(int));
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "debug_int", debug_int_cb);

                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "batchNorm2D_filterMean", batchNorm2D_filterMean);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D_Mean, threads, 1, 1);

                            print("channelSize: " + net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1]);
                            float[] filterMean = new float[net.layers[l].in_channels];
                            batchNorm2D_filterMean.GetData(filterMean);
                            Noedify_Utils.PrintArray(filterMean, "filterMean lyr " + l + ": ");

                            
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "batchNorm2D_filterMean", batchNorm2D_filterMean);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "batchNorm2D_filterStd", batchNorm2D_filterStd);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D_Std1, threads, 1, 1);

                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std2, "batchNorm2D_filterStd", batchNorm2D_filterStd);
                            threads = Mathf.CeilToInt((float)net.layers[l].in_channels / (float)4);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D_Std2, threads, 1, 1);

                            _shader.SetInt("apply_bias_layerSize", net.layers[l].layerSize);
                            _shader.SetInt("apply_bias_param_offset", net.activeNodeIdx_start[l - 1]);
                            _shader.SetInt("apply_bias_node_offset", net.nodeIdx_start[l]);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_out", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_param", net.biasMask_cbuf);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)8);
                            _shader.Dispatch(kernel_ApplyBias, threads, 1, 1);

                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "weights", net.networkWeights_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "batchNorm2D_filterMean", batchNorm2D_filterMean);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "batchNorm2D_filterStd", batchNorm2D_filterStd);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D, threads, 1, 1);
                            */
#endregion
                            break;
                        }
                    case (Noedify.LayerType.BatchNorm3D):
                        {

                            float[] filterMean = new float[net.layers[l].in_channels];
                            float[] filterVar = new float[net.layers[l].in_channels];

                            float[] batchNorm_outputs_prevlyr = new float[net.layers[l - 1].layerSize];
                            nodeOutputs_cbuf.GetData(batchNorm_outputs_prevlyr, 0, net.nodeIdx_start[l - 1], net.layers[l - 1].layerSize);
                            float[] batchNorm_inputs = new float[net.layers[l].layerSize];

                            // Noedify_Utils.PrintArray(batchNorm_inputs, "BatchNorm inputs");

                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                //print("filterMean[" + layer.conv3DLayer.filterTrack[j] + "](j=" + j + ") += " + inputs[j] + " = " + (filterMean[layer.conv3DLayer.filterTrack[j]] + inputs[j]));
                                filterMean[net.layers[l].conv3DLayer.filterTrack[j]] += batchNorm_outputs_prevlyr[j];
                            }

                            //print("filtermean: " + filterMean[0] + " / " + (layer.layerSize3D[0] * layer.layerSize3D[1]) + " + " + filterMean[0]);

                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                filterMean[c] /= (net.layers[l].layerSize3D[0] * net.layers[l].layerSize3D[1] * net.layers[l].layerSize3D[2]);
                            /*
                            string outString = "filterMean: ";
                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                outString += filterMean[c] + ", ";
                            print(outString);
                            */
                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                int filter_no = net.layers[l].conv3DLayer.filterTrack[j];
                                float varTemp = (batchNorm_outputs_prevlyr[j] - filterMean[filter_no]);
                                filterVar[filter_no] += varTemp * varTemp;
                            }
                            /*
                            string outStringVar = "filterVar: ";
                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                outStringVar += filterVar[c] + ", ";
                            print(outStringVar);
                            */
                            for (int c = 0; c < net.layers[l].in_channels; c++)
                                filterVar[c] = Mathf.Sqrt(filterVar[c] / ((net.layers[l].layerSize3D[0] * net.layers[l].layerSize3D[1] * net.layers[l].layerSize3D[2])) + net.layers[l].bn_eps);

                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                int filter_no = net.layers[l].conv3DLayer.filterTrack[j];
                                float inputHat = (batchNorm_outputs_prevlyr[j] - filterMean[filter_no]) / filterVar[filter_no];
                                batchNorm_inputs[j] = net.layers[l].weights.values[0, filter_no] * inputHat + net.layers[l].biases.values[filter_no];
                            }

                            nodeInputs_cbuf.SetData(batchNorm_inputs, 0, net.nodeIdx_start[l], net.layers[l].layerSize);

#region BathNormGPU
                            /*
                            ComputeBuffer batchNorm2D_filterMean = new ComputeBuffer(net.layers[l].in_channels, sizeof(float));
                            ComputeBuffer batchNorm2D_filterStd = new ComputeBuffer(net.layers[l].in_channels, sizeof(float));

                            ComputeBuffer debug_int_cb = new ComputeBuffer(1, sizeof(int));
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "debug_int", debug_int_cb);

                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Mean, "batchNorm2D_filterMean", batchNorm2D_filterMean);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D_Mean, threads, 1, 1);

                            print("channelSize: " + net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1]);
                            float[] filterMean = new float[net.layers[l].in_channels];
                            batchNorm2D_filterMean.GetData(filterMean);
                            Noedify_Utils.PrintArray(filterMean, "filterMean lyr " + l + ": ");

                            
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "batchNorm2D_filterMean", batchNorm2D_filterMean);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std1, "batchNorm2D_filterStd", batchNorm2D_filterStd);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D_Std1, threads, 1, 1);

                            _shader.SetBuffer(kernel_I2D_BatchNorm2D_Std2, "batchNorm2D_filterStd", batchNorm2D_filterStd);
                            threads = Mathf.CeilToInt((float)net.layers[l].in_channels / (float)4);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D_Std2, threads, 1, 1);

                            _shader.SetInt("apply_bias_layerSize", net.layers[l].layerSize);
                            _shader.SetInt("apply_bias_param_offset", net.activeNodeIdx_start[l - 1]);
                            _shader.SetInt("apply_bias_node_offset", net.nodeIdx_start[l]);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_out", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_ApplyBias, "apply_bias_param", net.biasMask_cbuf);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)8);
                            _shader.Dispatch(kernel_ApplyBias, threads, 1, 1);

                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "weights", net.networkWeights_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "filterTrack", net.filterTrack_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "nodeInputs", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "nodeOutputs", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "batchNorm2D_filterMean", batchNorm2D_filterMean);
                            _shader.SetBuffer(kernel_I2D_BatchNorm2D, "batchNorm2D_filterStd", batchNorm2D_filterStd);
                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_I2D_BatchNorm2D, threads, 1, 1);
                            */
#endregion
                            break;
                        }
                    case (Noedify.LayerType.FullyConnected):
                    case (Noedify.LayerType.Output):
                        {
                            /*
                            _shader.SetInt("n", net.layers[l-1].layerSize);
                            _shader.SetInt("k", net.layers[l].layerSize);
                            _shader.SetInt("m", 1);
                            _shader.SetInt("matA_startIndex", net.nodeIdx_start_l0_cb[l-1]);
                            _shader.SetInt("result_startIndex", net.nodeIdx_start_l0_cb[l]);

                            ComputeBuffer weights = new ComputeBuffer(net.layers[l].no_weights, sizeof(float));
                            ComputeBuffer biases = new ComputeBuffer(net.layers[l].no_biases, sizeof(float));
         
                            weights.SetData(net.networ, net.weightIdx_start_cb[l-1], 0, net.layers[l].no_weights);
                            biases.SetData(net.networkBiases_cb, net.biasIdx_start_cb[l-1], 0, net.layers[l].no_biases);
                            _shader.SetBuffer(kernel_MatMul, "matB", weights);
                            _shader.SetBuffer(kernel_MatMul, "matA", nodeOutputs_cbuf);
                            _shader.SetBuffer(kernel_MatMul, "offset", biases);
                            _shader.SetBuffer(kernel_MatMul, "result", nodeInputs_cbuf);

                            threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                            _shader.Dispatch(kernel_MatMul, threads, 1, 1);

                            float[] new_nodeInputs = new float[layerSize];
                            nodeInputs_cbuf.GetData(new_nodeInputs, 0, net.nodeIdx_start_l0_cb[l], layerSize);

                            _shader.SetInt("AF_offset", net.nodeIdx_start_l0_cb[l]);
                            _shader.SetInt("AF_type", (int)net.layers[l].activationFunction);

                            _shader.SetBuffer(kernel_AF, "result", nodeInputs_cbuf);
                            _shader.SetBuffer(kernel_AF, "matA", nodeOutputs_cbuf);
                            _shader.Dispatch(kernel_AF, threads, 1, 1);

                            weights.Dispose();
                            biases.Dispose();
                            */
                            break;
                        }
                    default:
                        {
                            //_shader.Dispatch(kernel, layerSize, 1, 1);
                            break;
                        }
                }

                _shader.SetInt("AF_type", (int)net.layers[l].activationFunction);
                _shader.SetBuffer(kernel_AF, "nodeInputs", nodeInputs_cbuf);
                _shader.SetBuffer(kernel_AF, "nodeOutputs", nodeOutputs_cbuf);
                threads = Mathf.CeilToInt((float)net.layers[l].layerSize / (float)16);
                _shader.Dispatch(kernel_AF, threads, 1, 1);

                UnityEngine.Profiling.Profiler.EndSample();
                sw1.Stop();
                Debug.Log("***Layer " + l + " (" + net.layers[l].name + ") evaluation time: " + sw1.ElapsedMilliseconds);
                //ApplyActivationFunction(ref nodeOutputs, nodeInputs, batch_nodes + nodeIndeces_start[layer_no], layerSize[layer_no], activationFunction);
            }
            /*
            sw.Stop();
            print("processing time: " + sw.ElapsedMilliseconds + " ms");
            sw.Reset();
            sw.Start();
            */
            /*
            float[] allOutputs = new float[net.total_no_nodes];
            float[] allInputs = new float[net.total_no_nodes];
            nodeOutputs_cbuf.GetData(allOutputs, 0, 0, net.total_no_nodes);
            nodeInputs_cbuf.GetData(allInputs, 0, 0, net.total_no_nodes);
            System.IO.StreamWriter writer = new System.IO.StreamWriter("Assets/Resources/nodeValues.txt", false);
            writer.WriteLine("node\tinput\toutput");
            int current_lyr = 1;
            writer.WriteLine("----- lyr 0 -----");
            for (int i = 0; i < net.total_no_nodes; i++)
            {
                if (i == net.nodeIdx_start[current_lyr])
                {
                    writer.WriteLine("----- lyr " + current_lyr + " -----");
                    current_lyr++;
                    if (current_lyr > (net.LayerCount()-1))
                        current_lyr = 0;
                }
                writer.WriteLine(i + ":\t" + allInputs[i] + "\t" + allOutputs[i]) ;
            }
            writer.Close();
            */
            Error:
            int predictionLayer = lastLayer;
            if (lastLayer < 0)
                predictionLayer = net.LayerCount() - 1;
            prediction = new float[net.layers[predictionLayer].layerSize];
            nodeOutputs_cbuf.GetData(prediction, 0, net.nodeIdx_start[predictionLayer], prediction.Length);

            CleanupTrainingCBuffers();

            net.trainingInProgress = false;
            evaluationInProgress = false;
            //sw.Stop();

        }
        else
        {
            if (!suppressMessages) print("Error: Evaluation initiated before previous job competed");
            evaluationInProgress = false;
            net.Cleanup_Par();
        }
        net.trainingInProgress = false;

        return;
    }

#if NOEDIFY_NORELEASE
    public void StopTraining()
    {
        if (train_coroutine != null)
            StopCoroutine(train_coroutine);
        activeJob.Complete();
    }
#endif

#if NOEDIFY_NOTRAIN
    Noedify.NN_LayerGradients[] AccumulateGradients(Noedify.Net net, float trainingRate, List<Noedify.NN_LayerGradients[]> batchGradients, int batch_size, float[] weighting)
    {
        Noedify.NN_LayerGradients[] net_gradients = new Noedify.NN_LayerGradients[net.LayerCount() - 1];
        for (int l = 1; l < net.LayerCount(); l++)
        {
            if (l >= fineTuningLayerLimit)
            {
                // Sum gradients across batch
                Noedify.NN_LayerGradients net_gradients_layer = new Noedify.NN_LayerGradients(net.layers, l);
                if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
                {
                    for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                    {
                        for (int j = 0; j < net.layers[l].conv2DLayer.N_weights_per_filter; j++)
                        {
                            for (int batch = 0; batch < batch_size; batch++)
                                net_gradients_layer.weight_gradients.valuesConv[f, j] += batchGradients[batch][l - 1].weight_gradients.valuesConv[f, j] * trainingRate * weighting[batch] / batch_size;
                        }
                        for (int batch = 0; batch < batch_size; batch++)
                            net_gradients_layer.bias_gradients.valuesConv[f] += batchGradients[batch][l - 1].bias_gradients.valuesConv[f] * trainingRate * weighting[batch] / batch_size;
                    }
                }
                else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D) { }
                else
                {
                    for (int j = 0; j < net.layers[l].layerSize; j++)
                    {
                        for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                            for (int batch = 0; batch < batch_size; batch++)
                                net_gradients_layer.weight_gradients.values[i, j] += batchGradients[batch][l - 1].weight_gradients.values[i, j] * trainingRate * weighting[batch] / batch_size;
                        for (int batch = 0; batch < batch_size; batch++)
                            net_gradients_layer.bias_gradients.values[j] += batchGradients[batch][l - 1].bias_gradients.values[j] * trainingRate * weighting[batch] / batch_size;
                    }
                }
                net_gradients[l - 1] = net_gradients_layer;
            }
        }
        return net_gradients;
    }

    Noedify.NN_LayerGradients[] AccumulateGradients(Noedify.Net net, float trainingRate, NativeArray<float> weightBatchGradients, NativeArray<float> biasBatchGradients, int batch_size, float[] weighting)
    {
        Noedify.NN_LayerGradients[] netGradients = new Noedify.NN_LayerGradients[net.LayerCount() - 1];

        for (int l = 1; l < net.LayerCount(); l++)
        {
            if (l >= fineTuningLayerLimit)
            {
                netGradients[l - 1] = new Noedify.NN_LayerGradients(net.layers, l);
                for (int batch = 0; batch < batch_size; batch++)
                {
                    if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
                    {
                        for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                        {
                            for (int j = 0; j < net.layers[l].conv2DLayer.N_weights_per_filter; j++)
                            {
                                netGradients[l - 1].weight_gradients.valuesConv[f, j] += weightBatchGradients[batch * net.total_no_weights + net.weightIdx_start[l - 1] + f * net.layers[l].conv2DLayer.N_weights_per_filter + j] * trainingRate * weighting[batch] / batch_size;
                            }
                            netGradients[l - 1].bias_gradients.valuesConv[f] += biasBatchGradients[batch * net.total_no_biases + net.biasIdx_start[l - 1] + f] * trainingRate * weighting[batch] / batch_size;
                        }
                    }
                    else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D) { }
                    else
                    {
                        for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                        {
                            for (int j = 0; j < net.layers[l].layerSize; j++)
                                netGradients[l - 1].weight_gradients.values[i, j] += weightBatchGradients[batch * net.total_no_weights + net.weightIdx_start[l - 1] + i * net.layers[l].layerSize + j] * trainingRate * weighting[batch] / batch_size;
                        }
                        for (int j = 0; j < net.layers[l].layerSize; j++)
                            netGradients[l - 1].bias_gradients.values[j] += biasBatchGradients[batch * net.total_no_biases + net.biasIdx_start[l - 1] + j] * trainingRate * weighting[batch] / batch_size;
                    }
                }
            }
        }
        return netGradients;
    }
#endif
#if NOEDIFY_NORELEASE
    public static void EvaluateLayer_Job(
        int layer_no,
        Noedify.LayerType layerType,
        Noedify.ActivationFunction activationFunction,
        Noedify.PoolingType poolingType,
        NativeArray<int> layerSize,
        ref NativeArray<float> nodeInputs,
        ref NativeArray<float> nodeOutputs,
        NativeArray<float> networkWeights,
        NativeArray<float> networkBiases,
        NativeArray<int> connectionMask,
        NativeArray<int> nodeIndeces_start,
        NativeArray<int> weightIndeces_start,
        NativeArray<int> biasIndeces_start,
        NativeArray<int> connectionMaskStart_index,
        NativeArray<int> CNN_nodes_per_filter,
        NativeArray<int> CNN_weights_per_filter,
        NativeArray<int> CNN_conns_per_node,
        int batch_nodes = 0)
    {

        if (layerType == Noedify.LayerType.Convolutional2D)
        {
            for (int j = 0; j < layerSize[layer_no]; j++)
            {
                int connsMade = 0;
                nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = 0;
                int f = Mathf.FloorToInt(j / CNN_nodes_per_filter[layer_no - 1]); // what filter/kernal is node j in
                for (int i = 0; i < layerSize[layer_no - 1]; i++)
                {
                    int connectionMask_j = connectionMask[connectionMaskStart_index[layer_no - 1] + i * layerSize[layer_no] + j];
                    if (connectionMask_j >= 0)
                    {
                        connsMade++;
                        nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] += nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i] * networkWeights[weightIndeces_start[layer_no - 1] + f * CNN_weights_per_filter[layer_no - 1] + connectionMask_j];
                    }
                    if (connsMade >= CNN_conns_per_node[layer_no - 1])
                        break;
                }
                nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] += networkBiases[biasIndeces_start[layer_no - 1] + f];
            }
        }
        else if (layerType == Noedify.LayerType.Pool2D)
        {

            for (int j = 0; j < layerSize[layer_no]; j++)
            {
                int connsMade = 0;
                if (poolingType == Noedify.PoolingType.Max)
                    nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = -100;
                else if (poolingType == Noedify.PoolingType.Avg)
                    nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = 0;

                for (int i = 0; i < layerSize[layer_no - 1]; i++)
                {
                    int connectionMask_j = connectionMask[connectionMaskStart_index[layer_no - 1] + i * layerSize[layer_no] + j];
                    if (connectionMask_j >= 0)
                    {
                        connsMade++;
                        if (poolingType == Noedify.PoolingType.Avg)
                            nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] += nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i];
                        else if (poolingType == Noedify.PoolingType.Max)
                        {
                            if (nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i] > nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j])
                            {
                                nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i];
                            }
                        }
                    }
                    if (connsMade >= CNN_conns_per_node[layer_no - 1])
                        break;
                }
                if (poolingType == Noedify.PoolingType.Avg)
                    nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] /= CNN_weights_per_filter[layer_no - 1];
            }

        }
        else
        {
            for (int j = 0; j < layerSize[layer_no]; j++)
            {
                nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] = 0;
                for (int i = 0; i < layerSize[layer_no - 1]; i++)
                {
                    nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] += nodeOutputs[batch_nodes + nodeIndeces_start[layer_no - 1] + i] * networkWeights[weightIndeces_start[layer_no - 1] + i * layerSize[layer_no] + j];
                }
                nodeInputs[batch_nodes + nodeIndeces_start[layer_no] + j] += networkBiases[biasIndeces_start[layer_no - 1] + j];
            }
        }
        ApplyActivationFunction(ref nodeOutputs, nodeInputs, batch_nodes + nodeIndeces_start[layer_no], layerSize[layer_no], activationFunction);
    }
#endif
#if NOEDIFY_NOTRAIN
    static void BackPropagateLayer_Job(
        int thisLayer,
        int nextLayer,
        int outputLayerNo,
        int batch,
        int total_no_active_nodes,
        int total_no_nodes,
        Noedify.ActivationFunction activationFunction,
        Noedify.PoolingType nextLayerPoolingType,
        NativeArray<Noedify.LayerType> layerTypes,
        NativeArray<int> layerSize,
        NativeArray<int> connectionMask,
        NativeArray<int> activeNodeIndeces_start,
        NativeArray<int> nodeIndeces_start,
        NativeArray<float> nodeInputs,
        NativeArray<float> nodeOutputs,
        NativeArray<float> networkWeights,
        NativeArray<int> weightIndeces_start,
        NativeArray<int> connectionMaskStart_index,
        NativeArray<int> CNN_nodes_per_filter,
        NativeArray<int> CNN_weights_per_filter,
        NativeArray<int> CNN_conns_per_node,
        ref NativeArray<float> deltas)
    {
        for (int i = 0; i < layerSize[thisLayer]; i++)
        {
            deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + i] = 0;

            for (int j = 0; j < layerSize[nextLayer]; j++)
            {
                if (layerTypes[nextLayer] == Noedify.LayerType.Convolutional2D)
                {
                    int connectionMask_j = connectionMask[connectionMaskStart_index[nextLayer - 1] + i * layerSize[nextLayer] + j];
                    if (connectionMask_j >= 0)
                    {
                        int f = Mathf.FloorToInt(j / CNN_nodes_per_filter[nextLayer - 1]); // what filter/kernal is node j in
                        deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + i] += networkWeights[weightIndeces_start[nextLayer - 1] + f * CNN_weights_per_filter[nextLayer - 1] + connectionMask_j] * deltas[batch * total_no_active_nodes + activeNodeIndeces_start[nextLayer - 1] + j];
                    }
                }
                else if (layerTypes[nextLayer] == Noedify.LayerType.Pool2D)
                {
                    int connectionMask_j = connectionMask[connectionMaskStart_index[nextLayer - 1] + i * layerSize[nextLayer] + j];
                    if (connectionMask_j >= 0)
                    {
                        if (nextLayerPoolingType == Noedify.PoolingType.Avg)
                            deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + i] += deltas[batch * total_no_active_nodes + activeNodeIndeces_start[nextLayer - 1] + j] / CNN_weights_per_filter[nextLayer - 1];
                        else if (nextLayerPoolingType == Noedify.PoolingType.Max)
                            if (nodeOutputs[batch * total_no_nodes + nodeIndeces_start[thisLayer] + i] == nodeInputs[batch * total_no_nodes + nodeIndeces_start[nextLayer] + j])
                                deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + i] += deltas[batch * total_no_active_nodes + activeNodeIndeces_start[nextLayer - 1] + j];
                    }


                }
                else
                    deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + i] += networkWeights[weightIndeces_start[nextLayer - 1] + i * layerSize[nextLayer] + j] * deltas[batch * total_no_active_nodes + activeNodeIndeces_start[nextLayer - 1] + j];

            }
            //deltas[batch * total_no_active_nodes + activeNodeIndeces_start[l - 1] + i] *= ApplyDeltaActivationFunction(nodeInputs[batch * total_no_nodes + nodeIndeces_start[l] + i], activationFunction[l]);
        }
        if (layerTypes[thisLayer] == Noedify.LayerType.Convolutional2D)
        {
            NativeArray<float> delta_bp_z = new NativeArray<float>(layerSize[thisLayer], Allocator.Temp);
            ApplyDeltaActivationFunction(ref delta_bp_z, 0, nodeInputs, batch * total_no_nodes + nodeIndeces_start[thisLayer], layerSize[thisLayer], activationFunction);
            for (int j = 0; j < layerSize[thisLayer]; j++)
                deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + j] *= delta_bp_z[j];
        }
    }

    static void CalculateGradientsLayer_Job(
        int thisLayer,
        int batch,
        int total_no_active_nodes,
        int total_no_nodes,
        int weights_arraySize,
        int biases_arraySize,
        Noedify.LayerType layerType,
        NativeArray<int> layerSize,
        NativeArray<float> nodeOutputs,
        ref NativeArray<float> networkWeightGradients,
        ref NativeArray<float> networkBiasGradients,
        NativeArray<int> connectionMask,
        NativeArray<int> activeNodeIndeces_start,
        NativeArray<int> nodeIndeces_start,
        NativeArray<int> weightIndeces_start,
        NativeArray<int> biasIndeces_start,
        NativeArray<int> connectionMaskStart_index,
        NativeArray<int> CNN_nodes_per_filter,
        NativeArray<int> CNN_weights_per_filter,
        NativeArray<int> CNN_conns_per_node,
        NativeArray<float> deltas)
    {
        int prevLayer = thisLayer - 1;
        for (int j = 0; j < layerSize[thisLayer]; j++)
        {
            int connsMade = 0;
            int f = 0;
            if (layerType == Noedify.LayerType.Convolutional2D)
                f = Mathf.FloorToInt(j / CNN_nodes_per_filter[thisLayer - 1]); // what filter/kernal is node j in
            for (int i = 0; i < layerSize[prevLayer]; i++)
            {
                if (layerType == Noedify.LayerType.Convolutional2D)
                {
                    int connectionMask_j = connectionMask[connectionMaskStart_index[thisLayer - 1] + i * layerSize[thisLayer] + j];
                    if (connectionMask_j >= 0)
                    {
                        connsMade++;
                        networkWeightGradients[batch * weights_arraySize + weightIndeces_start[thisLayer - 1] + f * CNN_weights_per_filter[thisLayer - 1] + connectionMask_j] += deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + j] * nodeOutputs[batch * total_no_nodes + nodeIndeces_start[prevLayer] + i];
                    }
                }
                else if (layerType == Noedify.LayerType.Pool2D) { break; }
                else
                    networkWeightGradients[batch * weights_arraySize + weightIndeces_start[thisLayer - 1] + i * layerSize[thisLayer] + j] = deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + j] * nodeOutputs[batch * total_no_nodes + nodeIndeces_start[prevLayer] + i];

                if (layerType == Noedify.LayerType.Convolutional2D)
                    if (connsMade >= CNN_nodes_per_filter[thisLayer - 1])
                        break;
            }
            //print("calculating bias gradients");
            if (layerType == Noedify.LayerType.Convolutional2D)
            {
                networkBiasGradients[batch * biases_arraySize + biasIndeces_start[thisLayer - 1] + f] += deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + j];
            }
            else if (layerType == Noedify.LayerType.Pool2D) { break; }
            else
                networkBiasGradients[batch * biases_arraySize + biasIndeces_start[thisLayer - 1] + j] = deltas[batch * total_no_active_nodes + activeNodeIndeces_start[thisLayer - 1] + j];
        }
    }
#endif
#if NOEDIFY_NORELEASE
    static void InitializeNetTempNativeArrays(
        Noedify.Net net,
        ref NativeArray<Noedify.ActivationFunction> activationFunctionList,
        ref NativeArray<Noedify.LayerType> layerTypeList,
        ref NativeArray<int> layerSizeList,
        ref NativeArray<int> CNN_nodes_per_filter_par,
        ref NativeArray<int> CNN_weights_per_filter_par,
        ref NativeArray<int> CNN_no_filters_par,
        ref NativeArray<int> CNN_conns_per_node,
        ref NativeArray<Noedify.PoolingType> poolingTypeList)
    {
        for (int l = 1; l < net.LayerCount(); l++)
        {
            if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
            {
                CNN_nodes_per_filter_par[l - 1] = net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1];
                CNN_weights_per_filter_par[l - 1] = net.layers[l].conv2DLayer.N_weights_per_filter;
                CNN_no_filters_par[l - 1] = net.layers[l].conv2DLayer.no_filters;
                CNN_conns_per_node[l - 1] = net.layers[l].conv2DLayer.N_connections_per_node;
            }
            else if (net.layers[l].layer_type == Noedify.LayerType.Pool2D)
            {
                poolingTypeList[l] = net.layers[l].pool_type;
                CNN_weights_per_filter_par[l - 1] = net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1];
                CNN_conns_per_node[l - 1] = net.layers[l].conv2DLayer.N_connections_per_node;
            }
            else
            {
                CNN_nodes_per_filter_par[l - 1] = 0;
                CNN_weights_per_filter_par[l - 1] = 0;
                CNN_no_filters_par[l - 1] = 0;
                CNN_conns_per_node[l - 1] = 0;
            }
        }

        for (int l = 0; l < net.LayerCount(); l++)
        {
            activationFunctionList[l] = net.layers[l].activationFunction;
            layerTypeList[l] = net.layers[l].layer_type;
            layerSizeList[l] = net.layers[l].layerSize;
        }
    }
#endif
    static void InitializeNetCBuffers(
    Noedify.Net net,
    ref List<object> cleanupList,
    ref ComputeBuffer activationFunctionList_cbuf,
    ref ComputeBuffer layerTypeList_cbuf,
    ref ComputeBuffer layerSizeList_cbuf,
    ref ComputeBuffer CNN_nodes_per_filter_cbuf,
    ref ComputeBuffer CNN_weights_per_filter_cbuf,
    ref ComputeBuffer CNN_no_filters_cbuf,
    ref ComputeBuffer CNN_conns_per_node_cbuf,
    ref ComputeBuffer poolingTypeList_cbuf)
    {
        int[] activationFunctionList_temp = new int[net.LayerCount()];
        int[] layerTypeList_temp = new int[net.LayerCount()];
        int[] layerSizeList_temp = new int[net.LayerCount()];
        int[] CNN_nodes_per_filter_temp = new int[net.LayerCount() - 1];
        int[] CNN_weights_per_filter_temp = new int[net.LayerCount() - 1];
        int[] CNN_no_filters_temp = new int[net.LayerCount() - 1];
        int[] CNN_conns_per_node_temp = new int[net.LayerCount() - 1];
        int[] poolingTypeList_temp = new int[net.LayerCount() - 1];

        for (int l = 1; l < net.LayerCount(); l++)
        {
            switch (net.layers[l].layer_type)
            {
                case (Noedify.LayerType.Convolutional2D):
                    {
                        CNN_nodes_per_filter_temp[l - 1] = net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1];
                        CNN_weights_per_filter_temp[l - 1] = net.layers[l].conv2DLayer.N_weights_per_filter;
                        CNN_no_filters_temp[l - 1] = net.layers[l].conv2DLayer.no_filters;
                        CNN_conns_per_node_temp[l - 1] = net.layers[l].conv2DLayer.N_connections_per_node;
                        break;
                    }
                case (Noedify.LayerType.Pool2D):
                    {
                        poolingTypeList_temp[l] = (int)net.layers[l].pool_type;
                        CNN_weights_per_filter_temp[l - 1] = net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1];
                        CNN_conns_per_node_temp[l - 1] = net.layers[l].conv2DLayer.N_connections_per_node;
                        break;
                    }
                case (Noedify.LayerType.TranspConvolutional2D):
                    {
                        CNN_nodes_per_filter_temp[l - 1] = net.layers[l].layerSize2D[0] * net.layers[l].layerSize2D[1];
                        CNN_weights_per_filter_temp[l - 1] = net.layers[l].conv2DLayer.N_weights_per_filter;
                        CNN_no_filters_temp[l - 1] = net.layers[l].conv2DLayer.no_filters;
                        CNN_conns_per_node_temp[l - 1] = net.layers[l].conv2DLayer.N_connections_per_node;
                        break;
                    }
                case (Noedify.LayerType.BatchNorm2D):
                    {
                        CNN_nodes_per_filter_temp[l - 1] = 0;
                        CNN_weights_per_filter_temp[l - 1] = 0;
                        CNN_no_filters_temp[l - 1] = net.layers[l].conv2DLayer.no_filters;
                        CNN_conns_per_node_temp[l - 1] = 0;
                        break;
                    }
                default:
                    {
                        CNN_nodes_per_filter_temp[l - 1] = 0;
                        CNN_weights_per_filter_temp[l - 1] = 0;
                        CNN_no_filters_temp[l - 1] = 0;
                        CNN_conns_per_node_temp[l - 1] = 0;
                        break;
                    }
            }
        }
        for (int l = 0; l < net.LayerCount(); l++)
        {
            activationFunctionList_temp[l] = (int)net.layers[l].activationFunction;
            layerTypeList_temp[l] = (int)net.layers[l].layer_type;
            layerSizeList_temp[l] = net.layers[l].layerSize;
        }


        activationFunctionList_cbuf.SetData(activationFunctionList_temp);
        layerTypeList_cbuf.SetData(layerTypeList_temp);
        layerSizeList_cbuf.SetData(layerSizeList_temp);
        CNN_nodes_per_filter_cbuf.SetData(CNN_nodes_per_filter_temp);
        CNN_weights_per_filter_cbuf.SetData(CNN_weights_per_filter_temp);
        CNN_no_filters_cbuf.SetData(CNN_no_filters_temp);
        CNN_conns_per_node_cbuf.SetData(CNN_conns_per_node_temp);
        poolingTypeList_cbuf.SetData(poolingTypeList_temp);

    }
#if NOEDIFY_NOTRAIN
    static float Cost(float[] networkOutputs, float[] expectedOutputs, CostFunction costFunction)
    {
        float cost = 0;
        int output_layer_size = networkOutputs.Length;
        switch (costFunction)
        {
            case (CostFunction.MeanSquare):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
#if NOEDIFY_MATHEMATICS
                        cost += 0.5f * Unity.Mathematics.math.pow(expectedOutputs[j] - networkOutputs[j], 2.0f);
#else
                        cost += 0.5f * Mathf.Pow(expectedOutputs[j] - networkOutputs[j], 2.0f);
#endif
                    }
                    break;
                }
            case (CostFunction.CrossEntropy):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
#if NOEDIFY_MATHEMATICS
                        cost += -1f / (float)output_layer_size * (expectedOutputs[j] * Unity.Mathematics.math.log(networkOutputs[j]) + ((1 - expectedOutputs[j]) * Unity.Mathematics.math.log(1f - networkOutputs[j])));
#else
                        cost += -1f / (float)output_layer_size * (expectedOutputs[j] * Mathf.Log(networkOutputs[j]) + ((1 - expectedOutputs[j]) * Mathf.Log(1f - networkOutputs[j])));
#endif
                    }
                    break;
                }
            default:
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
#if NOEDIFY_MATHEMATICS
                        cost += 0.5f * Unity.Mathematics.math.pow(expectedOutputs[j] - networkOutputs[j], 2.0f);
#else
                        cost += 0.5f * Mathf.Pow(expectedOutputs[j] - networkOutputs[j], 2.0f);
#endif
                    }
                    break;
                }
        }
        return cost;
    }
    static float Cost(NativeArray<float> networkOutputs, int networkOutputsStart_index, NativeArray<float> expectedOutputs, int expectedOutputsStart_index, int output_layer_size, CostFunction costFunction)
    {
        float cost = 0;

        switch (costFunction)
        {
            case (CostFunction.MeanSquare):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
#if NOEDIFY_MATHEMATICS
                        cost += 0.5f * Unity.Mathematics.math.pow(expectedOutputs[expectedOutputsStart_index + j] - networkOutputs[networkOutputsStart_index + j], 2.0f);
#else
                        cost += 0.5f * Mathf.Pow(expectedOutputs[expectedOutputsStart_index + j] - networkOutputs[networkOutputsStart_index + j], 2.0f);
#endif
                    }
                    break;
                }
            case (CostFunction.CrossEntropy):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
#if NOEDIFY_MATHEMATICS
                        cost += -1f / (float)output_layer_size * (expectedOutputs[expectedOutputsStart_index + j] * Unity.Mathematics.math.log(networkOutputs[networkOutputsStart_index + j]) + ((1 - expectedOutputs[expectedOutputsStart_index + j]) * Unity.Mathematics.math.log(1f - networkOutputs[networkOutputsStart_index + j])));
#else
                        cost += -1f / (float)output_layer_size * (expectedOutputs[expectedOutputsStart_index + j] * Mathf.Log(networkOutputs[networkOutputsStart_index + j]) + ((1 - expectedOutputs[expectedOutputsStart_index + j]) * Mathf.Log(1f - networkOutputs[networkOutputsStart_index + j])));
#endif
                    }
                    break;
                }
            default:
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
#if NOEDIFY_MATHEMATICS
                        cost += 0.5f * Unity.Mathematics.math.pow(expectedOutputs[expectedOutputsStart_index + j] - networkOutputs[networkOutputsStart_index + j], 2.0f);
#else
                        cost += 0.5f * Mathf.Pow(expectedOutputs[expectedOutputsStart_index + j] - networkOutputs[networkOutputsStart_index + j], 2.0f);
#endif
                    }
                    break;
                }
        }
        return cost;
    }

    static float[] DeltaCost(float[] networkOutputs, float[] expectedOutputs, CostFunction costFunction)
    {
        if (networkOutputs.Length != expectedOutputs.Length)
            print("ERROR: network output size (" + networkOutputs.Length + ") does not match training data label size (" + expectedOutputs.Length + ")");
        int output_layer_size = networkOutputs.Length;
        float[] deltaCost = new float[output_layer_size];
        switch (costFunction)
        {
            case (CostFunction.MeanSquare):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
                        deltaCost[j] = expectedOutputs[j] - networkOutputs[j];
                    }
                    break;
                }
            case (CostFunction.CrossEntropy):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
                        deltaCost[j] = 1f / (float)output_layer_size * (expectedOutputs[j] / (0.001f + networkOutputs[j]) + (expectedOutputs[j] - 1) / (1 - 0.001f - networkOutputs[j]));
                    }
                    break;
                }
            default:
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
                        deltaCost[j] = expectedOutputs[j] - networkOutputs[j];
                    }
                    break;
                }
        }
        return deltaCost;
    }
    static void DeltaCost(ref NativeArray<float> outputDeltas, int deltaStart_index, NativeArray<float> networkOutputs, int networkOutputsStart_index, NativeArray<float> expectedOutputs, int expectedOutputsStart_index, int output_layer_size, CostFunction costFunction)
    {

        switch (costFunction)
        {
            case (CostFunction.MeanSquare):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
                        outputDeltas[deltaStart_index + j] = expectedOutputs[expectedOutputsStart_index + j] - networkOutputs[networkOutputsStart_index + j];
                    }
                    break;
                }
            case (CostFunction.CrossEntropy):
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
                        outputDeltas[deltaStart_index + j] = 1f / (float)output_layer_size * (expectedOutputs[expectedOutputsStart_index + j] / (0.001f + networkOutputs[networkOutputsStart_index + j]) + (expectedOutputs[expectedOutputsStart_index + j] - 1) / (1 - 0.001f - networkOutputs[networkOutputsStart_index + j]));
                    }
                    break;
                }
            default:
                {
                    for (int j = 0; j < output_layer_size; j++)
                    {
                        outputDeltas[deltaStart_index + j] = expectedOutputs[expectedOutputsStart_index + j] - networkOutputs[networkOutputsStart_index + j];
                    }
                    break;
                }
        }
    }
#endif
    static float[] ApplyActivationFunction(float[] inputs, Noedify.ActivationFunction activationFunction)
    {
        float[] outputs = new float[inputs.Length];
        float normSum = 0;
        for (int j = 0; j < inputs.Length; j++)
        {
            switch (activationFunction)
            {
                case (Noedify.ActivationFunction.Sigmoid): outputs[j] = Sigmoid(inputs[j]); break;
                case (Noedify.ActivationFunction.ReLU):
                    {
                        if (inputs[j] < 0) outputs[j] = 0.00001f;
                        else outputs[j] = inputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.LeakyReLU):
                    {
                        if (inputs[j] < 0) outputs[j] = 0.05f * inputs[j];
                        else outputs[j] = inputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.Linear): outputs[j] = inputs[j]; break;
                case (Noedify.ActivationFunction.SoftMax):
                    {
                        outputs[j] = Mathf.Exp(inputs[j]);
                        normSum += outputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.ELU):
                    {
                        if (inputs[j] > 0)
                            outputs[j] = inputs[j];
                        else
                            outputs[j] = Mathf.Exp(inputs[j]) - 1;
                        break;
                    }
                case (Noedify.ActivationFunction.Hard_sigmoid):
                    {
                        if (inputs[j] > -2.5f)
                            outputs[j] = 0;
                        else if (inputs[j] > 2.5f)
                            outputs[j] = 1;
                        else
                            outputs[j] = 0.2f * inputs[j] + 0.5f;
                        break;
                    }
                case (Noedify.ActivationFunction.Tanh): outputs[j] = Tanh(inputs[j]); break;

                default: outputs[j] = 1.0f / (1.0f + Mathf.Exp(-inputs[j])); break;
            }
        }
        if (activationFunction == Noedify.ActivationFunction.SoftMax)
        {
            for (int j = 0; j < inputs.Length; j++)
                outputs[j] /= normSum + 0.001f;
        }
        return outputs;
    }
    static void ApplyActivationFunction(ref NativeArray<float> outputs, NativeArray<float> inputs, int layerStartIndex, int layerSize, Noedify.ActivationFunction activationFunction)
    {
        float normSum = 0;
        for (int j = layerStartIndex; j < (layerStartIndex + layerSize); j++)
        {
            switch (activationFunction)
            {
                case (Noedify.ActivationFunction.Sigmoid): outputs[j] = Sigmoid(inputs[j]); break;
                case (Noedify.ActivationFunction.ReLU):
                    {
                        if (inputs[j] < 0) outputs[j] = 0.00001f;
                        else outputs[j] = inputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.LeakyReLU):
                    {
                        if (inputs[j] < 0) outputs[j] = 0.05f * inputs[j];
                        else outputs[j] = inputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.Linear): outputs[j] = inputs[j]; break;
                case (Noedify.ActivationFunction.SoftMax):
                    {
#if NOEDIFY_MATHEMATICS
                        outputs[j] = Unity.Mathematics.math.exp(inputs[j]);
#else
                        outputs[j] = Mathf.Exp(inputs[j]);
#endif
                        normSum += outputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.ELU):
                    {
                        if (inputs[j] > 0)
                            outputs[j] = inputs[j];
                        else
                        {
#if NOEDIFY_MATHEMATICS
                            outputs[j] = Unity.Mathematics.math.exp(inputs[j]) - 1;
#else
                            outputs[j] = Mathf.Exp(inputs[j]) - 1;
#endif
                        }
                        break;
                    }
                case (Noedify.ActivationFunction.Hard_sigmoid):
                    {
                        if (inputs[j] > -2.5f)
                            outputs[j] = 0;
                        else if (inputs[j] > 2.5f)
                            outputs[j] = 1;
                        else
                            outputs[j] = 0.2f * inputs[j] + 0.5f;
                        break;
                    }
                case (Noedify.ActivationFunction.Tanh): outputs[j] = Tanh(inputs[j]); break;

#if NOEDIFY_MATHEMATICS
                default: outputs[j] = 1.0f / (1.0f + Unity.Mathematics.math.exp(-inputs[j])); break;
#else
                default: outputs[j] = 1.0f / (1.0f + Mathf.Exp(-inputs[j])); break;
#endif
            }
        }
        if (activationFunction == Noedify.ActivationFunction.SoftMax)
        {
            for (int j = layerStartIndex; j < (layerStartIndex + layerSize); j++)
                outputs[j] /= normSum + 0.001f;
        }
    }

    static float[] ApplyDeltaActivationFunction(float[] inputs, Noedify.ActivationFunction activationFunction)
    {
        float[] outputs = new float[inputs.Length];
        float normSum = 0;
        for (int j = 0; j < inputs.Length; j++)
        {
            switch (activationFunction)
            {
                case (Noedify.ActivationFunction.Sigmoid):
                    {
                        outputs[j] = Sigmoid(inputs[j]) * (1 - Sigmoid(inputs[j]));
                        break;
                    }
                case (Noedify.ActivationFunction.ReLU):
                    {
                        if (inputs[j] < 0) outputs[j] = 0.00001f;
                        else outputs[j] = 1;
                        break;
                    }
                case (Noedify.ActivationFunction.LeakyReLU):
                    {
                        if (inputs[j] < 0) outputs[j] = 0.05f;
                        else outputs[j] = 1;
                        break;
                    }
                case (Noedify.ActivationFunction.Linear): outputs[j] = 1; break;
                case (Noedify.ActivationFunction.SoftMax):
                    {
                        outputs[j] = Mathf.Exp(inputs[j]);
                        normSum += outputs[j];
                        break;
                    }
                case (Noedify.ActivationFunction.ELU):
                    {
                        if (inputs[j] > 0)
                            outputs[j] = 1;
                        else
                            outputs[j] = Mathf.Exp(inputs[j]);
                        break;
                    }
                case (Noedify.ActivationFunction.Hard_sigmoid):
                    {
                        if (inputs[j] > -2.5f)
                            outputs[j] = 0;
                        else if (inputs[j] > 2.5f)
                            outputs[j] = 0;
                        else
                            outputs[j] = 0.2f;
                        break;
                    }
                case (Noedify.ActivationFunction.Tanh): outputs[j] = 1 - Mathf.Pow(Tanh(inputs[j]), 2); break;

                default: outputs[j] = Sigmoid(inputs[j]) * (1 - Sigmoid(inputs[j])); break;
            }
        }
        if (activationFunction == Noedify.ActivationFunction.SoftMax)
        {
            float normSum_sq = normSum * normSum + 0.001f;
            for (int j = 0; j < inputs.Length; j++)
                outputs[j] = outputs[j] * (normSum - outputs[j]) / normSum_sq;
        }
        return outputs;
    }
    static void ApplyDeltaActivationFunction(ref NativeArray<float> outputs, int outputStartIndex, NativeArray<float> inputs, int inputStartIndex, int layerSize, Noedify.ActivationFunction activationFunction)
    {
        float normSum = 0;
        for (int j = 0; j < layerSize; j++)
        {
            switch (activationFunction)
            {
                case (Noedify.ActivationFunction.Sigmoid):
                    {
                        outputs[outputStartIndex + j] = Sigmoid(inputs[inputStartIndex + j]) * (1 - Sigmoid(inputs[inputStartIndex + j]));
                        break;
                    }
                case (Noedify.ActivationFunction.ReLU):
                    {
                        if (inputs[inputStartIndex + j] < 0) outputs[outputStartIndex + j] = 0.00001f;
                        else outputs[outputStartIndex + j] = 1;
                        break;
                    }
                case (Noedify.ActivationFunction.LeakyReLU):
                    {
                        if (inputs[inputStartIndex + j] < 0) outputs[outputStartIndex + j] = 0.05f;
                        else outputs[outputStartIndex + j] = 1;
                        break;
                    }
                case (Noedify.ActivationFunction.Linear): outputs[outputStartIndex + j] = 1; break;
                case (Noedify.ActivationFunction.SoftMax):
                    {
#if NOEDIFY_MATHEMATICS
                        outputs[outputStartIndex + j] = Unity.Mathematics.math.exp(inputs[inputStartIndex + j]);
#else
                        outputs[outputStartIndex + j] = Mathf.Exp(inputs[inputStartIndex + j]);
#endif
                        normSum += outputs[outputStartIndex + j];
                        break;
                    }
                case (Noedify.ActivationFunction.ELU):
                    {
                        if (inputs[inputStartIndex + j] > 0)
                            outputs[outputStartIndex + j] = 1;
                        else
                        {
#if NOEDIFY_MATHEMATICS
                            outputs[outputStartIndex + j] = Unity.Mathematics.math.exp(inputs[inputStartIndex + j]);
#else
                            outputs[outputStartIndex + j] = Mathf.Exp(inputs[inputStartIndex + j]);
#endif
                        }
                        break;
                    }
                case (Noedify.ActivationFunction.Hard_sigmoid):
                    {
                        if (inputs[inputStartIndex + j] > -2.5f)
                            outputs[outputStartIndex + j] = 0;
                        else if (inputs[inputStartIndex + j] > 2.5f)
                            outputs[outputStartIndex + j] = 0;
                        else
                            outputs[outputStartIndex + j] = 0.2f;
                        break;
                    }
#if NOEDIFY_MATHEMATICS
                case (Noedify.ActivationFunction.Tanh): outputs[outputStartIndex + j] = 1 - Unity.Mathematics.math.pow(Tanh(inputs[inputStartIndex + j]), 2); break;
#else
                case (Noedify.ActivationFunction.Tanh): outputs[outputStartIndex + j] = 1 - Mathf.Pow(Tanh(inputs[inputStartIndex + j]), 2); break;
#endif
                default: outputs[outputStartIndex + j] = Sigmoid(inputs[inputStartIndex + j]) * (1 - Sigmoid(inputs[inputStartIndex + j])); break;
            }
        }
        if (activationFunction == Noedify.ActivationFunction.SoftMax)
        {
            float normSum_sq = normSum * normSum + 0.001f;
            for (int j = 0; j < layerSize; j++)
                outputs[outputStartIndex + j] = outputs[outputStartIndex + j] * (normSum - outputs[outputStartIndex + j]) / normSum_sq;
        }
    }

#if NOEDIFY_NORELEASE
    NativeArray<float> NativeAllocFloat(int size, Allocator allocator = Allocator.Persistent)
    {
        NativeArray<float> newArray = new NativeArray<float>(size, allocator);
        nativeArrayCleanupList.Add(newArray);
        return newArray;
    }
    NativeArray<int> NativeAllocInt(int size, Allocator allocator = Allocator.Persistent)
    {
        NativeArray<int> newArray = new NativeArray<int>(size, allocator);
        nativeArrayCleanupList.Add(newArray);
        return newArray;
    }
#endif

    void CleanupTrainingNativeArrays()
    {
        if (nativeArrayCleanupList != null)
        {
            if (nativeArrayCleanupList.Count > 0)
            {
                for (int i = 0; i < nativeArrayCleanupList.Count; i++)
                {
                    if (nativeArrayCleanupList[i] is NativeArray<int>)
                        ((NativeArray<int>)nativeArrayCleanupList[i]).Dispose();
                    else if (nativeArrayCleanupList[i] is NativeArray<float>)
                        ((NativeArray<float>)nativeArrayCleanupList[i]).Dispose();
                    else if (nativeArrayCleanupList[i] is NativeArray<Noedify.ActivationFunction>)
                        ((NativeArray<Noedify.ActivationFunction>)nativeArrayCleanupList[i]).Dispose();
                    else if (nativeArrayCleanupList[i] is NativeArray<Noedify.LayerType>)
                        ((NativeArray<Noedify.LayerType>)nativeArrayCleanupList[i]).Dispose();
                    else if (nativeArrayCleanupList[i] is NativeArray<Noedify.PoolingType>)
                        ((NativeArray<Noedify.PoolingType>)nativeArrayCleanupList[i]).Dispose();
                    else
                    {
                        if (!suppressMessages) print("ERROR (CleanupTrainingNativeArrays): Unknown memory object type " + nativeArrayCleanupList[i].ToString());
                    }
                }
            }
        }
        nativeArrayCleanupList = new List<object>();
    }

    void CleanupTrainingCBuffers()
    {
        if (cBufferCleanupList != null)
        {
            if (cBufferCleanupList.Count > 0)
            {
                for (int i = 0; i < cBufferCleanupList.Count; i++)
                {
                    if (cBufferCleanupList[i] is ComputeBuffer)
                    {
                        ((ComputeBuffer)cBufferCleanupList[i]).Dispose();
                    }
                    else
                    {
                        if (!suppressMessages) print("ERROR (CleanupTrainingCBuffers): Unknown memory object type " + cBufferCleanupList[i].ToString());
                    }
                }
            }
        }
        cBufferCleanupList = new List<object>();
    }

    void SetCShaderBuffer(ComputeShader _shader, int kernel, Noedify.Net net,
        ComputeBuffer activationFunctionList_cbuf,
        ComputeBuffer layerTypeList_cbuf,
        ComputeBuffer layerSizeList_cbuf,
        ComputeBuffer CNN_nodes_per_filter_cbuf,
        ComputeBuffer CNN_weights_per_filter_cbuf,
        ComputeBuffer CNN_no_filters_cbuf,
        ComputeBuffer CNN_conns_per_node_cbuf,
        ComputeBuffer poolingTypeList_cbuf,
        ComputeBuffer netInputs_cbuf,
        ComputeBuffer netOutputs_cbuf,
        ComputeBuffer nodeOutputs_cbuf,
        ComputeBuffer nodeInputs_cbuf
        )
    {

        _shader.SetBuffer(kernel, "activationFunctionList", activationFunctionList_cbuf);
        _shader.SetBuffer(kernel, "layerTypeList", layerTypeList_cbuf);
        _shader.SetBuffer(kernel, "layerSizeList", layerSizeList_cbuf);
        _shader.SetBuffer(kernel, "CNN_nodes_per_filter", CNN_nodes_per_filter_cbuf);
        _shader.SetBuffer(kernel, "CNN_weights_per_filter", CNN_weights_per_filter_cbuf);
        _shader.SetBuffer(kernel, "CNN_no_filters", CNN_no_filters_cbuf);
        _shader.SetBuffer(kernel, "CNN_conns_per_node", CNN_conns_per_node_cbuf);
        _shader.SetBuffer(kernel, "poolingTypeList", poolingTypeList_cbuf);

        //_shader.SetBuffer(kernel, "netInputs", netInputs_cbuf);
        //_shader.SetBuffer(kernel, "netOutputs", netOutputs_cbuf);
        _shader.SetBuffer(kernel, "nodeOutputs", nodeOutputs_cbuf);
        _shader.SetBuffer(kernel, "nodeInputs", nodeInputs_cbuf);
        /*
        _shader.SetBuffer(kernel, "weights", net.networkWeights_cbuf);
        _shader.SetBuffer(kernel, "biases", net.networkBiases_cbuf);
        _shader.SetBuffer(kernel, "weightId", net.weightId_start_cbuf);
        _shader.SetBuffer(kernel, "biasId", net.biasId_start_cbuf);
        _shader.SetBuffer(kernel, "activeNodeId", net.activeNodeId_start_cbuf);
        _shader.SetBuffer(kernel, "nodeId", net.nodeId_start_l0_cbuf);
        _shader.SetBuffer(kernel, "connectionMask", net.connectionMask_cbuf);
        _shader.SetBuffer(kernel, "connectionMaskId", net.connectionMaskId_start_cbuf);
        _shader.SetBuffer(kernel, "connections", net.connections_cbuf);
        _shader.SetBuffer(kernel, "connetionInFilter", net.connectionsInFilter_cbuf);
        _shader.SetBuffer(kernel, "connectionsId", net.connectionsId_start_cbuf);
        */

    }

    void OnApplicationQuit()
    {
#if NOEDIFY_NOTRAIN
        StopTraining();
#endif
        CleanupTrainingNativeArrays();
        CleanupTrainingCBuffers();
        if (nets_cleanup_list.Count > 0)
        {
            for (int i = 0; i < nets_cleanup_list.Count; i++)
            {
                if (nets_cleanup_list[i] != null)
                {
                    if (nets_cleanup_list[i].nativeArraysInitialized)
                        nets_cleanup_list[i].Cleanup_Par();
                    if (nets_cleanup_list[i].cBuffersInitialized)
                        nets_cleanup_list[i].Cleanup_ComputeBuffers();
                }
            }
        }
    }

    static float Tanh(float x)
    {
        if (x < -50)
            return -1.0f;
        else if (x > 50.0f)
            return 1.0f;

#if NOEDIFY_MATHEMATICS
        return (Unity.Mathematics.math.exp(2.0f * x) - 1.0f) / (Unity.Mathematics.math.exp(2.0f * x) + 1.0f);
#else
        return (float)(Mathf.Exp(2.0f * x) - 1.0f) / (float)(Mathf.Exp(2.0f * x) + 1.0f);
#endif
    }

    static float Sigmoid(float x)
    {
#if NOEDIFY_MATHEMATICS
        return 1.0f / (1.0f + Unity.Mathematics.math.exp(-x));
#else
        return 1.0f / (1.0f + Mathf.Exp(-x));
#endif
    }
}
