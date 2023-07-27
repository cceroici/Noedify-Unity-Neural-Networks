//#define CONST_INIT_STATE
//#define NOEDIFY_NORELEASE
#define PYTORCH_MODE

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using UnityEditor;

public class Noedify : MonoBehaviour
{
    public enum LayerType
    {
        Input, Output, FullyConnected,
        Input2D, Output2D,
        Convolutional2D, Pool2D,
        Input3D, Output3D,
        TranspConvolutional2D, BatchNorm2D, ActivationFunction, ActivationFunction2D,
        Convolutional3D, BatchNorm3D, TranspConvolutional3D,
    };
    public enum ActivationFunction { Sigmoid, ReLU, LeakyReLU, Linear, SoftMax, ELU, Hard_sigmoid, Tanh };

    public enum PoolingType { Max, Avg };

    [System.Serializable]
    public class Layer
    {
        public LayerType layer_type;
        public string name;
        public int layerSize;
        public int[] layerSize2D;
        public int[] layerSize3D;
        public int in_channels;
        public int layer_no;
        public bool trainingActive;
        // BatchNorm parameters
        public float bn_eps;
        public bool bn_running_track;
        public float[] bn_running_mean;
        public float[] bn_running_var;
        public NN_Weights weights;
        public NN_Biases biases;
        public ActivationFunction activationFunction;
        public PoolingType pool_type;
        public int no_weights;
        public int no_biases;

        public Noedify_Convolutional2D.Convolutional2DLayer conv2DLayer;
        public Noedify_Convolutional3D.Convolutional3DLayer conv3DLayer;

        #region Layer Constructors
        // Input 1D
        public Layer(LayerType newType, int newLayerSize, string newLayerName = "")
        {
            if (newType != LayerType.Convolutional2D)
            {
                name = newLayerName;
                layer_type = newType;
                layerSize = newLayerSize;
                layerSize2D = new int[2];
                layerSize2D[0] = layerSize;
                layerSize2D[1] = 1;
                in_channels = 1;
            }
            else
                print("Error adding layer " + name);
        }

        // Input 2D, Input3D
        public Layer(LayerType newType, int[] inSize, int noChannels, string newLayerName = "")
        {
            name = newLayerName;
            in_channels = noChannels;
            if (newType == LayerType.Input2D)
            {
                layer_type = LayerType.Input2D;
                layerSize2D = inSize;
                layerSize = layerSize2D[0] * layerSize2D[1] * in_channels;
                conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                conv2DLayer.no_filters = in_channels;
            }
            else if (newType == LayerType.Input3D)
            {
                layer_type = LayerType.Input3D;
                layerSize3D = inSize;
                layerSize = layerSize3D[0] * layerSize3D[1] * layerSize3D[2] * in_channels;
                conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                conv3DLayer.no_filters = in_channels;
            }
            else
                print("Error adding layer " + name);
        }

        // Fully-connected hidden, output
        public Layer(LayerType newType, int newLayerSize, ActivationFunction actFunction, string newLayerName = "")
        {

            if (newType != LayerType.Convolutional2D)
            {
                name = newLayerName;
                layer_type = newType;
                layerSize = newLayerSize;
                layerSize2D = new int[2];
                layerSize2D[0] = layerSize;
                layerSize2D[1] = 1;
                in_channels = 1;
                activationFunction = actFunction;
            }
            else
                print("Error adding layer " + name);
        }

        // convolutional 2D, transpose convolutional 2D, convolutional 3D, transpose convolutional 3D
        public Layer(LayerType newType, Layer previousLayer, int[] filtsize, int[] strd, int nfilters, int[] pdding, ActivationFunction actFunction, string newLayerName = "")
        {
            if (newType == LayerType.Convolutional2D)
            {
                name = newLayerName;
                conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                in_channels = 1;
                layer_type = newType;
                conv2DLayer.stride = strd;
                activationFunction = actFunction;
                conv2DLayer.no_filters = nfilters;

                if (previousLayer.layer_type == LayerType.Input2D)
                    in_channels = previousLayer.in_channels;
                else
                    in_channels = previousLayer.conv2DLayer.no_filters;

                conv2DLayer.filterSize = filtsize;
                conv2DLayer.padding = pdding;
                layerSize2D = new int[2];
#if PYTORCH_MODE
                layerSize2D[0] = Mathf.FloorToInt(((float)previousLayer.layerSize2D[0] + 2f * (float)conv2DLayer.padding[0] - (float)conv2DLayer.filterSize[0]) / (float)conv2DLayer.stride[0] + 1f);
                layerSize2D[1] = Mathf.FloorToInt(((float)previousLayer.layerSize2D[1] + 2f * (float)conv2DLayer.padding[1] - (float)conv2DLayer.filterSize[1]) / (float)conv2DLayer.stride[1] + 1f);
#else
                layerSize2D[0] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[0] + 2f * (float)conv2DLayer.padding[0] - (float)conv2DLayer.filterSize[0]) / (float)conv2DLayer.stride[0] + 1f);
                layerSize2D[1] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[1] + 2f * (float)conv2DLayer.padding[1] - (float)conv2DLayer.filterSize[1]) / (float)conv2DLayer.stride[1] + 1f);
#endif
                layerSize = layerSize2D[0] * layerSize2D[1] * conv2DLayer.no_filters;
            }
            else if (newType == LayerType.TranspConvolutional2D)
            {
                name = newLayerName;
                conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                in_channels = nfilters;
                conv2DLayer.no_filters = nfilters;
                layer_type = newType;
                conv2DLayer.stride = strd;
                activationFunction = actFunction;

                conv2DLayer.filterSize = filtsize;
                conv2DLayer.padding = pdding;

                if (Noedify_Utils.Is1DLayerType(previousLayer))
                {

                    layerSize2D = filtsize;
                    conv2DLayer.no_filters = in_channels;
                    layerSize = layerSize2D[0] * layerSize2D[1] * conv2DLayer.no_filters;
                }
                else
                {
                    if (previousLayer.layer_type == LayerType.Input2D)
                        conv2DLayer.no_filters = nfilters * previousLayer.in_channels;
                    else
                    {
                        conv2DLayer.no_filters = nfilters;
                        in_channels = previousLayer.conv2DLayer.no_filters;
                    }

                    layerSize2D = new int[2];
                    layerSize2D[0] = Mathf.CeilToInt((previousLayer.layerSize2D[0] - 1) * conv2DLayer.stride[0] + conv2DLayer.filterSize[0] - 2 * conv2DLayer.padding[0]);
                    layerSize2D[1] = Mathf.CeilToInt((previousLayer.layerSize2D[1] - 1) * conv2DLayer.stride[1] + conv2DLayer.filterSize[1] - 2 * conv2DLayer.padding[1]);
                    layerSize = layerSize2D[0] * layerSize2D[1] * conv2DLayer.no_filters;

                    //print("sequential TC2D layer size: (" + layerSize2D[0] + "," + layerSize2D[1] + ")");
                    //print("prev layer size: (" + previousLayer.layerSize2D[0] + "," + previousLayer.layerSize2D[1] + ")");
                }
            }
            else if (newType == LayerType.Convolutional3D)
            {
                name = newLayerName;
                conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                in_channels = 1;
                layer_type = newType;
                conv3DLayer.stride = strd;
                activationFunction = actFunction;
                conv3DLayer.no_filters = nfilters;

                if (previousLayer.layer_type == LayerType.Input3D)
                    in_channels = previousLayer.in_channels;
                else
                    in_channels = previousLayer.conv3DLayer.no_filters;

                conv3DLayer.filterSize = filtsize;
                conv3DLayer.padding = pdding;
                layerSize3D = new int[3];
#if PYTORCH_MODE
                layerSize3D[0] = Mathf.FloorToInt(((float)previousLayer.layerSize3D[0] + 2f * (float)conv3DLayer.padding[0] - (float)conv3DLayer.filterSize[0]) / (float)conv3DLayer.stride[0] + 1f);
                layerSize3D[1] = Mathf.FloorToInt(((float)previousLayer.layerSize3D[1] + 2f * (float)conv3DLayer.padding[1] - (float)conv3DLayer.filterSize[1]) / (float)conv3DLayer.stride[1] + 1f);
                layerSize3D[2] = Mathf.FloorToInt(((float)previousLayer.layerSize3D[2] + 2f * (float)conv3DLayer.padding[2] - (float)conv3DLayer.filterSize[2]) / (float)conv3DLayer.stride[2] + 1f);
#else
                conv3DLayer[0] = Mathf.CeilToInt(((float)previousLayer.layerSize3D[0] + 2f * (float)conv3DLayer.padding[0] - (float)conv3DLayer.filterSize[0]) / (float)conv3DLayer.stride[0] + 1f);
                conv3DLayer[1] = Mathf.CeilToInt(((float)previousLayer.layerSize3D[1] + 2f * (float)conv3DLayer.padding[1] - (float)conv3DLayer.filterSize[1]) / (float)conv3DLayer.stride[1] + 1f);
#endif
                layerSize = layerSize3D[0] * layerSize3D[1] * layerSize3D[2] * conv3DLayer.no_filters;

            }
            else if (newType == LayerType.TranspConvolutional3D)
            {
                name = newLayerName;
                conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                in_channels = nfilters;
                conv3DLayer.no_filters = nfilters;
                layer_type = newType;
                conv3DLayer.stride = strd;
                activationFunction = actFunction;

                conv3DLayer.filterSize = filtsize;
                conv3DLayer.padding = pdding;

                if (Noedify_Utils.Is1DLayerType(previousLayer))
                {
                    layerSize3D = filtsize;
                    conv3DLayer.no_filters = in_channels;
                    layerSize = layerSize3D[0] * layerSize3D[1] * conv3DLayer.no_filters;
                }
                else
                {
                    if (previousLayer.layer_type == LayerType.Input3D)
                        conv3DLayer.no_filters = nfilters * previousLayer.in_channels;
                    else
                    {
                        conv3DLayer.no_filters = nfilters;
                        in_channels = previousLayer.conv3DLayer.no_filters;
                    }

                    layerSize3D = new int[3];
                    layerSize3D[0] = Mathf.CeilToInt((previousLayer.layerSize3D[0] - 1) * conv3DLayer.stride[0] + conv3DLayer.filterSize[0] - 2 * conv3DLayer.padding[0]);
                    layerSize3D[1] = Mathf.CeilToInt((previousLayer.layerSize3D[1] - 1) * conv3DLayer.stride[1] + conv3DLayer.filterSize[1] - 2 * conv3DLayer.padding[1]);
                    layerSize3D[2] = Mathf.CeilToInt((previousLayer.layerSize3D[2] - 1) * conv3DLayer.stride[2] + conv3DLayer.filterSize[2] - 2 * conv3DLayer.padding[2]);
                    layerSize = layerSize3D[0] * layerSize3D[1] * layerSize3D[2] * conv3DLayer.no_filters;

                    //print("sequential TC3D layer size: (" + layerSize3D[0] + "," + layerSize3D[1] + ")");
                    //print("prev layer size: (" + previousLayer.layerSize3D[0] + "," + previousLayer.layerSize23[1] + ")");
                }
            }
            else
                print("Error adding layer " + name);
        }

        // Pool2D
        public Layer(LayerType newType, Layer previousLayer, int[] shape, int[] strd, int[] pdding, PoolingType pooling_type, string newLayerName = "")
        {
            if (newType == LayerType.Pool2D)
            {
                //print("adding pool lyr");
                name = newLayerName;
                conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                in_channels = 1;
                layer_type = newType;
                conv2DLayer.stride = strd;
                pool_type = pooling_type;
                activationFunction = ActivationFunction.Linear;
                if (previousLayer.layer_type != LayerType.Convolutional2D)
                {
                    print("Error: Pool2D layer must appear after a convolutional2D layer");
                    goto Error;
                }
                conv2DLayer.filterSize[0] = shape[0];
                conv2DLayer.filterSize[1] = shape[1];
                conv2DLayer.padding = pdding;
                layerSize2D = new int[2];
                layerSize2D[0] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[0] + 2 * (float)conv2DLayer.padding[0] - (float)conv2DLayer.filterSize[0]) / (float)conv2DLayer.stride[0] + 1);
                layerSize2D[1] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[1] + 2 * (float)conv2DLayer.padding[1] - (float)conv2DLayer.filterSize[1]) / (float)conv2DLayer.stride[1] + 1);
                //conv2DLayer.padding[0] = layerSize2D[0] - Mathf.FloorToInt(((float)previousLayer.layerSize2D[0] - (float)conv2DLayer.filterSize[0]) / (float)conv2DLayer.stride[0] + 1);
                //conv2DLayer.padding[1] = layerSize2D[1] - Mathf.FloorToInt(((float)previousLayer.layerSize2D[1] - (float)conv2DLayer.filterSize[1]) / (float)conv2DLayer.stride[1] + 1);
                layerSize = layerSize2D[0] * layerSize2D[1] * previousLayer.conv2DLayer.no_filters;
                conv2DLayer.no_filters = previousLayer.conv2DLayer.no_filters;
            }
            else
            {
                print("Error adding layer " + newLayerName);
            }
            Error: { }

        }

        // BatchNorm2D, BatchNorm3D
        public Layer(LayerType newType, Layer previousLayer, float epsilon, ActivationFunction actFunction, bool running_track=true, string newLayerName = "")
        {
            if (newType.Equals(LayerType.BatchNorm2D))
            {
                if (Noedify_Utils.Is2DLayerType(previousLayer))
                {
                    layer_type = newType;
                    name = newLayerName;
                    bn_eps = epsilon;
                    bn_running_track = running_track;
                    conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                    conv2DLayer.filterSize = previousLayer.conv2DLayer.filterSize;
                    conv2DLayer.no_filters = previousLayer.conv2DLayer.no_filters;
                    activationFunction = actFunction;
                    in_channels = previousLayer.conv2DLayer.no_filters;
                    layerSize = previousLayer.layerSize;
                    layerSize2D = previousLayer.layerSize2D;
                    if (bn_running_track)
                    {
                        bn_running_mean = new float[in_channels];
                        bn_running_var = new float[in_channels];
                        for (int i=0; i<in_channels; i++){
                            bn_running_mean[i] = 0.0f;
                            bn_running_var[i] = 1.0f;
                        }
                    }
                }
                else
                {
                    print("Error: BatchNorm2D layer must appear after a 2D layer");
                    goto Error;
                }
            }
            if (newType.Equals(LayerType.BatchNorm3D))
            {
                if (Noedify_Utils.Is3DLayerType(previousLayer))
                {
                    layer_type = newType;
                    name = newLayerName;
                    bn_eps = epsilon;
                    bn_running_track = running_track;
                    conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                    conv3DLayer.filterSize = previousLayer.conv3DLayer.filterSize;
                    conv3DLayer.no_filters = previousLayer.conv3DLayer.no_filters;
                    activationFunction = actFunction;
                    in_channels = previousLayer.conv3DLayer.no_filters;
                    layerSize = previousLayer.layerSize;
                    layerSize3D = previousLayer.layerSize3D;
                    if (bn_running_track)
                    {
                        bn_running_mean = new float[in_channels];
                        bn_running_var = new float[in_channels];
                        for (int i=0; i<in_channels; i++){
                            bn_running_mean[i] = 0.0f;
                            bn_running_var[i] = 1.0f;
                        }
                    }
                }
                else
                {
                    print("Error: BatchNorm3D layer must appear after a 3D layer");
                    goto Error;
                }
            }
            Error: { }
        }

        // Output 2D
        public Layer(LayerType newType, Layer previousLayer, string newLayerName = "")
        {
            if (newType == LayerType.Output2D)
            {
                name = newLayerName;
                layer_type = newType;
                activationFunction = ActivationFunction.Linear;
                if (previousLayer.layer_type != LayerType.Convolutional2D & previousLayer.layer_type != LayerType.Pool2D)
                {
                    print("Error: Output2D layer must appear after a convolutional2D or Pool2D layer");
                    goto Error;
                }
                layerSize2D = previousLayer.layerSize2D;
                in_channels = previousLayer.conv2DLayer.no_filters;
            }
            else
            {
                print("Error adding layer " + newLayerName + ", incorrect layer type: " + newType.ToString() + ", expected: Output2D");
            }
            Error: { }

        }
    }
    #endregion
    [System.Serializable]
    public class InputArray
    {
        public float[,] array1;
        public float[,,] array2;
        public float[,,,] array3;
        public int[] dims;
        public bool set_array_1D;
        public bool set_array_2D;
        public bool set_array_3D;
        public int w;
        public int h;
        public int d;
        public int ch;

        public InputArray(float[,] new_array) // 1D
        {
            array1 = new_array;
            set_array_2D = true;
            w = new_array.GetLength(1);
            h = 1;
            d = 1;
        }

        public InputArray(float[,,] new_array) // 2D
        {
            array2 = new_array;
            set_array_2D = true;
            w = new_array.GetLength(2);
            h = new_array.GetLength(1);
            d = 1;
            ch = new_array.GetLength(0);
        }

        public InputArray(float[,,,] new_array) // 3D
        {
            array3 = new_array;
            set_array_3D = true;
            w = new_array.GetLength(1);
            h = new_array.GetLength(2);
            d = new_array.GetLength(3);
            ch = new_array.GetLength(0);
        }

        public float[] FlattenArray()
        {
            if (set_array_1D)
                return Noedify_Utils.FlattenDataset(array1);
            else if (set_array_2D)
                return Noedify_Utils.FlattenDataset(array2);
            else if (set_array_3D)
                return Noedify_Utils.FlattenDataset(array3);
            else
            {
                print("Error: InputArray.FlattenArray(): no input set");
                return null;
            }
        }

        public bool MatchesSize(int testSize, int test_ch_count, bool suppressOutput = false)
        {
            if (testSize == w && test_ch_count == ch)
                return true;
            else
            {
                if (!suppressOutput)
                    print("Input mismatch! Expected input size: [" + test_ch_count + ", " + testSize + "], received size: [" + ch + ", " + w + "]");
                return false;
            }
        }
        public bool MatchesSize(int[] testSize, int test_ch_count, bool suppressOutput = false)
        {
            if (testSize.Length == 2)
            {
                if (testSize[0] == w && testSize[1] == h && test_ch_count == ch)
                    return true;
            }
            else if (testSize.Length == 3)
            {
                if (testSize[0] == w && testSize[1] == h && testSize[2] == d && test_ch_count == ch)
                    return true;
            }

            if (!suppressOutput & testSize.Length == 2)
                print("Input mismatch! Expected input size: [" + test_ch_count + ", " + testSize[0] + ", " + testSize[1] + "], received size: [" + ch + ", " + w + ", " + h + "]");
            if (!suppressOutput & testSize.Length == 3)
                print("Input mismatch! Expected input size: [" + test_ch_count + ", " + testSize[0] + ", " + testSize[1] + ", " + testSize[2] + "], received size: [" + ch + ", " + w + ", " + h + ", " + d + "]");
            return false;
        }

    }

    [System.Serializable]

    #region Network Parameters
    public class NN_Weights
    {
        public int count;
        public int count_prevLyr;
        public float[,] values;
        public float[,] valuesConv;

        // 1D layers, BatchNorm2D, BatchNorm3D
        public NN_Weights(int newCount, int newCountPrevLyr, bool randomize)
        {
            count = newCount;
            count_prevLyr = newCountPrevLyr;
            values = new float[count_prevLyr, count];
            valuesConv = new float[0, 0];
            if (randomize)
            {
#if (CONST_INIT_STATE)
                Random.InitState(5);
#endif
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < count_prevLyr; j++)
                        values[j, i] = (Random.Range(0f, 10f) - 5f) / 20f;
                }
            }
        }
        // 2D/3D Convolutional, 2D/3D Transpose Convolutional
        public NN_Weights(Layer layer, Layer previousLayer, bool randomize)
        {
            count = layer.layerSize;
            valuesConv = new float[0, 0];
            values = new float[0, 0];
            switch (layer.layer_type)
            {
                case (Noedify.LayerType.Convolutional2D):

                    valuesConv = new float[layer.conv2DLayer.no_filters, layer.in_channels * layer.conv2DLayer.filterSize[1] * layer.conv2DLayer.filterSize[0]];

                    if (randomize)
                    {
#if (CONST_INIT_STATE)
                    Random.InitState(5);
#endif
                        for (int c = 0; c < layer.in_channels; c++)
                            for (int f = 0; f < layer.conv2DLayer.no_filters; f++)
                                for (int i = 0; i < layer.conv2DLayer.filterSize[1]; i++)
                                    for (int j = 0; j < layer.conv2DLayer.filterSize[0]; j++)
                                        valuesConv[f, c * (layer.conv2DLayer.filterSize[1] * layer.conv2DLayer.filterSize[0]) + j * layer.conv2DLayer.filterSize[0] + i] = (Random.Range(0f, 10f) - 5f) / 30f;
                    }
                    break;
                case (Noedify.LayerType.TranspConvolutional2D):
                    if (Noedify_Utils.Is1DLayerType(previousLayer)) // transpose Convolutional (previous layer 1D)
                    {
                        values = new float[previousLayer.layerSize, layer.layerSize];
                        if (randomize)
                        {
#if (CONST_INIT_STATE)
                    Random.InitState(5);
#endif
                            for (int i = 0; i < previousLayer.layerSize; i++)
                                for (int j = 0; j < layer.layerSize; j++)
                                    values[i, j] = (Random.Range(0f, 10f) - 5f) / 30f;
                        }
                    }
                    else // tranpose Convolutional (previous layer 2D)
                    {
                        valuesConv = new float[layer.conv2DLayer.no_filters, previousLayer.conv2DLayer.no_filters * layer.conv2DLayer.filterSize[1] * layer.conv2DLayer.filterSize[0]];

                        if (randomize)
                        {
#if (CONST_INIT_STATE)
                    Random.InitState(5);
#endif
                            for (int c = 0; c < previousLayer.conv2DLayer.no_filters; c++)
                                for (int f = 0; f < layer.conv2DLayer.no_filters; f++)
                                    for (int i = 0; i < layer.conv2DLayer.filterSize[1]; i++)
                                        for (int j = 0; j < layer.conv2DLayer.filterSize[0]; j++)
                                            valuesConv[f, c * (layer.conv2DLayer.filterSize[1] * layer.conv2DLayer.filterSize[0]) + j * layer.conv2DLayer.filterSize[0] + i] = (Random.Range(0f, 10f) - 5f) / 30f;
                        }
                    }


                    break;
                case (Noedify.LayerType.Convolutional3D):
                    valuesConv = new float[layer.conv3DLayer.no_filters, layer.in_channels * layer.conv3DLayer.filterSize[2] * layer.conv3DLayer.filterSize[1] * layer.conv3DLayer.filterSize[0]];
                    break;
                case (Noedify.LayerType.TranspConvolutional3D):
                    {
                        if (Noedify_Utils.Is1DLayerType(previousLayer)) // transpose Convolutional (previous layer 1D)
                        {
                            values = new float[previousLayer.layerSize, layer.layerSize];
                            if (randomize)
                            {
#if (CONST_INIT_STATE)
                    Random.InitState(5);
#endif
                                for (int i = 0; i < previousLayer.layerSize; i++)
                                    for (int j = 0; j < layer.layerSize; j++)
                                        values[i, j] = (Random.Range(0f, 10f) - 5f) / 30f;
                            }
                        }
                        else // tranpose Convolutional (previous layer 3D)
                        {
                            valuesConv = new float[layer.conv3DLayer.no_filters, previousLayer.conv3DLayer.no_filters * layer.conv3DLayer.filterSize[2] * layer.conv3DLayer.filterSize[1] * layer.conv3DLayer.filterSize[0]];

                            if (randomize)
                            {
#if (CONST_INIT_STATE)
                    Random.InitState(5);
#endif
                                for (int c = 0; c < previousLayer.conv3DLayer.no_filters; c++)
                                    for (int f = 0; f < layer.conv3DLayer.no_filters; f++)
                                        for (int i = 0; i < layer.conv3DLayer.filterSize[1]; i++)
                                            for (int j = 0; j < layer.conv3DLayer.filterSize[0]; j++)
                                                valuesConv[f, c * (layer.conv3DLayer.filterSize[1] * layer.conv3DLayer.filterSize[0]) + j * layer.conv3DLayer.filterSize[0] + i] = (Random.Range(0f, 10f) - 5f) / 30f;
                            }
                        }
                        break;
                    }
                default: print("Weight initialization failed, unknown layer type: " + layer.layer_type); break;
            }
        }

        public void TestInitialize()
        {
            if (valuesConv != null)
            {
                for (int f = 0; f < 4; f++)
                {
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            valuesConv[f, i * 3 + j] = 0;
                }
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < count_prevLyr; j++)
                    {
                        if (i == j)
                            values[j, i] = 1;
                        else
                            values[j, i] = 0;
                    }
                }
            }
        }
    }

    [System.Serializable]
    public class NN_Biases
    {
        public int count;
        public float[] values;
        public float[] valuesConv;

        public NN_Biases(int newCount, bool randomize)
        {
            count = newCount;
            values = new float[count];
            valuesConv = new float[0];
            if (randomize)
            {
#if (CONST_INIT_STATE)
                Random.InitState(5);
#endif

                for (int i = 0; i < count; i++)
                {
                    values[i] = (Random.Range(0f, 10f) - 5f) / 20f;
                }
            }
        }

        public NN_Biases(int newCount, int no_filters, bool randomize)
        {
            count = newCount;
            valuesConv = new float[no_filters];
            values = new float[0];
            if (randomize)
            {
#if (CONST_INIT_STATE)
                Random.InitState(5);
#endif
                for (int f = 0; f < no_filters; f++)
                    valuesConv[f] = Random.Range(0, 10f) / 10f;
            }
        }

        public void GaussianInitialization()
        {

        }

        public void TestInitialize()
        {
            if (valuesConv != null)
            {
                for (int f = 0; f < 4; f++)
                    valuesConv[f] = 0;
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    values[i] = 0;
                }
            }
        }
    }

    [System.Serializable]
    public class NN_LayerGradients
    {
        public int no_nodes;
        public NN_Weights weight_gradients;
        public NN_Biases bias_gradients;

        public NN_LayerGradients(List<Layer> layers, int layer_no)
        {
            no_nodes = layers[layer_no].layerSize;
            if (layers[layer_no].layer_type == LayerType.Convolutional2D)
            {
                weight_gradients = new NN_Weights(layers[layer_no], layers[layer_no - 1], false);
                bias_gradients = new NN_Biases(no_nodes, layers[layer_no].conv2DLayer.no_filters, false);
            }
            else
            {
                weight_gradients = new NN_Weights(no_nodes, layers[layer_no - 1].layerSize, false);
                bias_gradients = new NN_Biases(no_nodes, false);
            }
        }

    }

    #endregion

    [System.Serializable]
    public class Net
    {
        public List<Layer> layers;
        public bool trainingInProgress;

        public int total_no_nodes;
        public int total_no_activeNodes;
        public int total_no_weights;
        public int total_no_biases;

        public bool nativeArraysInitialized;

        public NativeArray<float> networkWeights_par;
        public NativeArray<float> networkBiases_par;
        public NativeArray<float> biasMask_par;

        public NativeArray<float> networkWeights_gradients_par;
        public NativeArray<float> networkBiases_gradients_par;
        public NativeArray<int> weightIdx_start;
        public NativeArray<int> biasIdx_start;
        public NativeArray<int> weightGradientIndeces_start;
        public NativeArray<int> biasGradientIndeces_start;
        public NativeArray<int> activeNodeIdx_start;
        public NativeArray<int> nodeIdx_start;
        public NativeArray<int> connectionMask_par;
        public NativeArray<int> connectionMaskIndeces_start;
        public NativeArray<int> connections_par;
        public NativeArray<int> connectionsInFilter_par;
        public NativeArray<int> connectionsIdx_start;
        public NativeArray<int> filterTrack_par;
        public NativeArray<int> nodeTrack_par;

        public bool cBuffersInitialized;
        public ComputeShader _shader;
        public ComputeBuffer networkWeights_cbuf;
        public ComputeBuffer networkBiases_cbuf;
        public ComputeBuffer biasMask_cbuf;
        //public NativeArray<float> networkWeights_gradients_par;
        //public NativeArray<float> networkBiases_gradients_par;
        public ComputeBuffer weightIdx_start_cbuf;
        public ComputeBuffer biasIdx_start_cbuf;
        //public int weightGradientIndeces_start_cbuf;
        //public NativeArray<int> biasGradientIndeces_start;
        public ComputeBuffer activeNodeIdx_start_cbuf;
        public ComputeBuffer nodeIdx_start_l0_cbuf;
        public ComputeBuffer connections_cbuf;
        public ComputeBuffer connectionsInFilter_cbuf;
        public ComputeBuffer filterTrack_cbuf;
        public ComputeBuffer nodeTrack_cbuf;
        public ComputeBuffer connectionsIdx_start_cbuf;


        public Net()
        {
            layers = new List<Layer>();
            nativeArraysInitialized = false;
        }

        public Layer AddLayer(Layer new_layer)
        {
            if (Noedify_Utils.IsInputLayerType(new_layer) & (LayerCount() != 0))
                print("Warning: Input layer can only be added first");
            else if (!Noedify_Utils.IsInputLayerType(new_layer) & (LayerCount() == 0))
                print("Warning: Input layer must be added first");
            else if (new_layer.layer_type.Equals(LayerType.Convolutional2D) & LayerCount() > 0)
            {
                if (!Noedify_Utils.Is2DLayerType(layers[LayerCount() - 1]))
                    Debug.LogError("Warning: Can only add convolutional layer after 2D layer");
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else if (new_layer.layer_type.Equals(LayerType.Pool2D))
            {
                if (layers[LayerCount() - 1].layer_type != LayerType.Convolutional2D)
                    print("Warning: Pool2D layer " + new_layer.name + " can only be added after a Convolutional2D layer.");
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else if (new_layer.layer_type.Equals(LayerType.TranspConvolutional2D))
            {
                if (layers[LayerCount() - 1].layer_type == LayerType.Output | layers[LayerCount() - 1].layer_type == LayerType.Output2D)
                    print("Warning: Transpose Convolutional 2D layer cannot be added after an output layer.");
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else if (new_layer.layer_type.Equals(LayerType.BatchNorm2D))
            {
                if (!Noedify_Utils.Is2DLayerType(layers[LayerCount() - 1]))
                    print("Warning: BatchNorm2D layer can only be added after a 2D layer");
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else if (new_layer.layer_type.Equals(LayerType.BatchNorm3D))
            {
                if (!Noedify_Utils.Is3DLayerType(layers[LayerCount() - 1]))
                    print("Warning: BatchNorm3D layer can only be added after a 3D layer");
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else if (new_layer.layer_type.Equals(LayerType.Convolutional3D))
            {
                if (LayerCount() == 0)
                    print("Warning: Adding Conv3D layer with no input layer");
                else if (!Noedify_Utils.Is3DLayerType(layers[LayerCount() - 1]))
                {
                    print("Warning: Conv3D layer " + LayerCount() + " has non-3D preceeding layer type: " + layers[LayerCount() - 1].layer_type.ToString());
                }
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else if (new_layer.layer_type.Equals(LayerType.TranspConvolutional3D))
            {
                if (LayerCount() == 0)
                    print("Warning: Adding Transpose Conv3D layer with no input layer");
                else if (layers[LayerCount() - 1].layer_type == LayerType.Output | layers[LayerCount() - 1].layer_type == LayerType.Output2D)
                    print("Warning: Transpose Convolutional 3D layer cannot be added after an output layer.");
                else if (!Noedify_Utils.Is3DLayerType(layers[LayerCount() - 1]))
                {
                    print("Warning: Transpose Conv3D layer " + LayerCount() + " has non-3D preceeding layer type: " + layers[LayerCount() - 1].layer_type.ToString());
                }
                else
                {
                    new_layer.layer_no = LayerCount();
                    layers.Add(new_layer);
                }
            }
            else
            {
                new_layer.layer_no = LayerCount();
                layers.Add(new_layer);
                print("added layer: " + new_layer.name);
            }
            return new_layer;
        }

        public void BuildNetwork(bool showSummary = false)
        {
            if (LayerCount() > 0)
            {
                int total_no_biases = Get_Total_No_Biases();
                int total_no_weights = Get_Total_No_Weights();


                print("Building network with " + (LayerCount() - 2) + " hidden layers, " + total_no_biases + " biases, and " + total_no_weights + " weights");
                CalculateLayerParameters();
                if (showSummary)
                    NetworkSummary();
                for (int l = 1; l < LayerCount(); l++)
                {
                    switch (layers[l].layer_type)
                    {
                        case (LayerType.FullyConnected):
                        case (LayerType.Output):
                            {
                                layers[l].weights = new NN_Weights(layers[l].layerSize, layers[l - 1].layerSize, true);
                                layers[l].biases = new NN_Biases(layers[l].layerSize, true);
                                break;
                            }
                        case (LayerType.Convolutional2D):
                            {
                                layers[l].conv2DLayer.BuildConnections(layers[l - 1], layers[l]);
                                layers[l].weights = new NN_Weights(layers[l], layers[l - 1], true);
                                layers[l].biases = new NN_Biases(layers[l].layerSize, layers[l].conv2DLayer.no_filters, true);
                                break;
                            }
                        case (LayerType.TranspConvolutional2D):
                            {
                                layers[l].conv2DLayer.BuildConnectionsTransConv2D(layers[l - 1], layers[l]);
                                layers[l].weights = new NN_Weights(layers[l], layers[l - 1], true);
                                layers[l].biases = new NN_Biases(layers[l].layerSize, layers[l].in_channels, true);

                                break;
                            }
                        case (LayerType.Pool2D):
                            {
                                layers[l].conv2DLayer.BuildConnectionsPool2D(layers[l - 1], layers[l]);
                                break;
                            }
                        case (LayerType.TranspConvolutional3D):
                            {
                                layers[l].conv3DLayer.BuildConnectionsTransConv3D(layers[l - 1], layers[l]);
                                layers[l].weights = new NN_Weights(layers[l], layers[l - 1], false);
                                layers[l].biases = new NN_Biases(layers[l].layerSize, layers[l].in_channels, false);
                                break;
                            }
                        case (LayerType.BatchNorm2D):
                            {
                                layers[l].conv2DLayer.BuildTrackers2D(layers[l]);
                                layers[l].weights = new NN_Weights(layers[l].in_channels, 1, false);
                                layers[l].biases = new NN_Biases(layers[l].in_channels, false);
                                layers[l].conv2DLayer.filterTrack = layers[l - 1].conv2DLayer.filterTrack;
                                layers[l].conv2DLayer.channelTrack = layers[l - 1].conv2DLayer.channelTrack;

                                for (int i = 0; i < layers[l].in_channels; i++)
                                {
                                    layers[l].weights.values[0, i] = 1;
                                    layers[l].biases.values[i] = 1;
                                }
                                break;
                            }
                        case (LayerType.BatchNorm3D):
                            {
                                layers[l].conv3DLayer.BuildTrackers3D(layers[l]);
                                layers[l].weights = new NN_Weights(layers[l].in_channels, 1, false);
                                layers[l].biases = new NN_Biases(layers[l].in_channels, false);
                                layers[l].conv3DLayer.filterTrack = layers[l - 1].conv3DLayer.filterTrack;
                                layers[l].conv3DLayer.channelTrack = layers[l - 1].conv3DLayer.channelTrack;

                                for (int i = 0; i < layers[l].in_channels; i++)
                                {
                                    layers[l].weights.values[0, i] = 1;
                                    layers[l].biases.values[i] = 1;
                                }
                                break;
                            }
                        case (LayerType.Convolutional3D):
                            {
                                layers[l].conv3DLayer.BuildConnections(layers[l - 1], layers[l]);
                                layers[l].weights = new NN_Weights(layers[l], layers[l - 1], false);
                                layers[l].biases = new NN_Biases(layers[l].layerSize, layers[l].conv3DLayer.no_filters, false);
                                break;
                            }
                        default: print("Warning: layer " + l + " unknown layer type '" + layers[l].layer_type.ToString() + "'"); break;

                    }

                }
                trainingInProgress = false;
            }
            else
                print("WARNING: BuildNetwork failed. Network is empty");
        }

        public void NetworkSummary()
        {
            print("Layer" + new string(' ', 35) + "Output Shape" + new string(' ', 26) + "# of weights" + new string(' ', 24) + "# of biases");
            const int spacerSize = 40;

            for (int l = 0; l < LayerCount(); l++)
            {
                string labelText = "(" + l + ") " + layers[l].name;
                string spacer1 = new string(' ', 40 - labelText.Length);

                string shapeText = "";
                string weightsText = "";
                string biasesText = "";

                switch (layers[l].layer_type)
                {
                    case (LayerType.FullyConnected):
                    case (LayerType.Input):
                    case (LayerType.Output):
                        {
                            shapeText = "[" + layers[l].layerSize + "]";
                            weightsText = layers[l].no_weights.ToString();
                            biasesText = layers[l].no_biases.ToString();
                            break;
                        }
                    case (LayerType.Input2D):
                        {
                            shapeText = "[" + layers[l].in_channels + ", " + layers[l].layerSize2D[0] + ", " + layers[l].layerSize2D[1] + "]";
                            weightsText = layers[l].no_weights.ToString();
                            biasesText = layers[l].no_biases.ToString();
                            break;
                        }
                    case (LayerType.Input3D):
                        {
                            shapeText = "[" + layers[l].in_channels + ", " + layers[l].layerSize3D[0] + ", " + layers[l].layerSize3D[1] + ", " + layers[l].layerSize3D[2] + "]";
                            weightsText = layers[l].no_weights.ToString();
                            biasesText = layers[l].no_biases.ToString();
                            break;
                        }
                    case (LayerType.Convolutional2D):
                    case (LayerType.Pool2D):
                    case (LayerType.TranspConvolutional2D):
                    case (LayerType.BatchNorm2D):
                        {
                            shapeText = "[" + layers[l].conv2DLayer.no_filters + ", " + layers[l].layerSize2D[0] + ", " + layers[l].layerSize2D[1] + "]";
                            weightsText = layers[l].no_weights.ToString();
                            biasesText = layers[l].no_biases.ToString();
                            break;
                        }
                    case (LayerType.BatchNorm3D):
                        {
                            shapeText = "[" + layers[l].conv3DLayer.no_filters + ", " + layers[l].layerSize3D[0] + ", " + layers[l].layerSize3D[1] + ", " + layers[l].layerSize3D[2] + "]";
                            weightsText = layers[l].no_weights.ToString();
                            biasesText = layers[l].no_biases.ToString();
                            break;
                        }
                    case (LayerType.Convolutional3D):
                    case (LayerType.TranspConvolutional3D):
                        {
                            shapeText = "[" + layers[l].conv3DLayer.no_filters + ", " + layers[l].layerSize3D[0] + ", " + layers[l].layerSize3D[1] + ", " + layers[l].layerSize3D[2] + "]";
                            weightsText = layers[l].no_weights.ToString();
                            biasesText = layers[l].no_biases.ToString();
                            break;
                        }
                    default: { print("Warning: (Noedify.Net.NetworkSummary()) Incompatable layer type: " + layers[l].layer_type.ToString() + " (l=" + l); break; }
                }

                string spacer2;
                string spacer3;
                if (shapeText.Length < spacerSize)
                    spacer2 = new string(' ', spacerSize - shapeText.Length);
                else
                    spacer2 = new string(' ', spacerSize);
                if (shapeText.Length < spacerSize)
                    spacer3 = new string(' ', spacerSize - weightsText.Length);
                else
                    spacer3 = new string(' ', spacerSize);

                Debug.Log(labelText + spacer1 + shapeText + spacer2 + weightsText + spacer3 + biasesText);
            }
        }

        // Calculate total # of parameters
        public void CalculateLayerParameters()
        {
            for (int l = 1; l < LayerCount(); l++)
            {
                switch (layers[l].layer_type)
                {
                    case (LayerType.FullyConnected):
                    case (LayerType.Output):
                        {
                            layers[l].no_weights = layers[l].layerSize * layers[l - 1].layerSize;
                            layers[l].no_biases = layers[l].layerSize;
                            break;
                        }
                    case (LayerType.Convolutional2D):
                        {
                            int input_channels = 1;
                            input_channels = layers[l - 1].conv2DLayer.no_filters;

                            layers[l].no_weights = layers[l].conv2DLayer.no_filters * input_channels * layers[l].conv2DLayer.filterSize[1] * layers[l].conv2DLayer.filterSize[0];
                            layers[l].no_biases = layers[l].conv2DLayer.no_filters;
                            break;
                        }
                    case (LayerType.Convolutional3D):
                        {
                            int input_channels = 1;
                            input_channels = layers[l - 1].conv3DLayer.no_filters;

                            layers[l].no_weights = layers[l].conv3DLayer.no_filters * input_channels * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.filterSize[0];
                            layers[l].no_biases = layers[l].conv3DLayer.no_filters;
                            break;
                        }
                    case (LayerType.Pool2D): break;
                    case (LayerType.TranspConvolutional2D):
                        {
                            if (Noedify_Utils.Is1DLayerType(layers[l - 1]))
                                layers[l].no_weights = layers[l - 1].layerSize * layers[l].layerSize;
                            else
                                layers[l].no_weights = layers[l].conv2DLayer.no_filters * layers[l].in_channels * layers[l].conv2DLayer.filterSize[1] * layers[l].conv2DLayer.filterSize[0];
                            layers[l].no_biases = layers[l].conv2DLayer.no_filters;
                            break;
                        }
                    case (LayerType.TranspConvolutional3D):
                        {
                            if (Noedify_Utils.Is1DLayerType(layers[l - 1]))
                                layers[l].no_weights = layers[l - 1].layerSize * layers[l].layerSize;
                            else
                                layers[l].no_weights = layers[l].conv3DLayer.no_filters * layers[l].in_channels * layers[l].conv3DLayer.filterSize[2] * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.filterSize[0];
                            layers[l].no_biases = layers[l].conv3DLayer.no_filters;
                            break;
                        }
                    case (LayerType.BatchNorm2D):
                        {
                            layers[l].no_weights = layers[l].conv2DLayer.no_filters;
                            layers[l].no_biases = layers[l].conv2DLayer.no_filters;
                            break;
                        }
                    case (LayerType.BatchNorm3D):
                        {
                            layers[l].no_weights = layers[l].conv3DLayer.no_filters;
                            layers[l].no_biases = layers[l].conv3DLayer.no_filters;
                            break;
                        }
                    default:
                        {
                            print("Warning: (Noedify.Net.CalculateLayerParameters()) Incompatable layer type: " + layers[l].layer_type.ToString() + " (l=" + l);
                            break;
                        }
                }

            }
        }

        public void SaveModel(string name, string dir = "")
        {
            Noedify_Manager.SavedModel saveModel = new Noedify_Manager.SavedModel(this, name, System.DateTime.Now);
            saveModel.modelName = name;
            saveModel.model = new Noedify_Manager.SerializedModel(this);
            saveModel.Save(dir);
        }

        public void SaveCompressedModel(string name, string dir = "")
        {
            Noedify_Manager.SavedCompressedModel saveModel = new Noedify_Manager.SavedCompressedModel(this, name, System.DateTime.Now);
            saveModel.modelName = name;
            saveModel.Save(dir);
        }

        public bool LoadModel(string name, string dir = "")
        {
            Net loadModel = Noedify_Manager.Load(name, dir);
            if (loadModel == null)
            {
                print("WARNING: Loading of " + name + " failed. File not found.");
                return false;
            }
            else
            {
                layers = loadModel.layers;
                return true;
            }
        }

        public bool LoadCompressedModel(string name, string dir = "")
        {
            Net loadModel = new Net();
            bool status = Noedify_Manager.LoadCompressedModel(ref loadModel, name, dir);
            total_no_nodes = loadModel.total_no_nodes;
            total_no_activeNodes = loadModel.total_no_activeNodes;
            total_no_weights = loadModel.total_no_weights;
            total_no_biases = loadModel.total_no_biases;
            layers = loadModel.layers;

            if (status == false)
            {
                print("WARNING: Loading of " + dir + name + " failed. File not found.");
                return false;
            }
            else
            {
                return true;
            }
        }

        public void ApplyGradients(NN_LayerGradients[] netGradients, int batch_size, bool par = false, int fineTuningLayerLimit = 0)
        {
            if (!par)
            {
                for (int l = 1; l < LayerCount(); l++)
                {
                    if (l >= fineTuningLayerLimit)
                    {
                        if (layers[l].layer_type == LayerType.Convolutional2D)
                        {
                            for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                            {
                                for (int j = 0; j < layers[l].conv2DLayer.N_weights_per_filter; j++)
                                {
                                    layers[l].weights.valuesConv[f, j] += netGradients[l - 1].weight_gradients.valuesConv[f, j];
                                }
                                layers[l].biases.valuesConv[f] += netGradients[l - 1].bias_gradients.valuesConv[f];
                            }
                        }
                        else if (layers[l].layer_type == LayerType.Pool2D) { }
                        else
                        {
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                for (int i = 0; i < layers[l - 1].layerSize; i++)
                                {
                                    layers[l].weights.values[i, j] += netGradients[l - 1].weight_gradients.values[i, j];
                                }
                                layers[l].biases.values[j] += netGradients[l - 1].bias_gradients.values[j];
                            }
                        }
                    }
                }
            }
            else
            {
                for (int l = 1; l < LayerCount(); l++)
                {
                    if (l >= fineTuningLayerLimit)
                    {
                        if (layers[l].layer_type == LayerType.Convolutional2D)
                        {
                            for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                            {
                                for (int j = 0; j < layers[l].conv2DLayer.N_weights_per_filter; j++)
                                {
                                    networkWeights_par[weightIdx_start[l - 1] + f * layers[l].conv2DLayer.N_weights_per_filter + j] += netGradients[l - 1].weight_gradients.valuesConv[f, j];
                                }
                                networkBiases_par[biasIdx_start[l - 1] + f] += netGradients[l - 1].bias_gradients.valuesConv[f];
                            }
                        }
                        else if (layers[l].layer_type == LayerType.Pool2D) { }
                        else
                        {
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                for (int i = 0; i < layers[l - 1].layerSize; i++)
                                    networkWeights_par[weightIdx_start[l - 1] + i * layers[l].layerSize + j] += netGradients[l - 1].weight_gradients.values[i, j];
                                networkBiases_par[biasIdx_start[l - 1] + j] += netGradients[l - 1].bias_gradients.values[j];
                            }
                        }
                    }
                }
            }
        }

        public void GenerateNativeParameterArrays()
        {
            total_no_nodes = 0;
            total_no_activeNodes = 0;
            total_no_weights = 0;
            total_no_biases = 0;
            int connections_no_nodes = 0;

            // Following arrays have size [LayerCount()-1]
            List<int> weightIdx_start_temp = new List<int>();
            List<int> biasIdx_start_temp = new List<int>();
            List<int> activeNodeIdx_start_temp = new List<int>();
            // Following arrays have size [LayerCount()]
            List<int> nodeIdx_start_temp = new List<int>();
            List<int> connectionsIdx_start_temp = new List<int>();

            for (int l = 1; l < LayerCount(); l++)
            {

                connectionsIdx_start_temp.Add(connections_no_nodes);
                weightIdx_start_temp.Add(total_no_weights);
                biasIdx_start_temp.Add(total_no_biases);

                switch (layers[l].layer_type)
                {
                    case (LayerType.Convolutional2D):
                        {

                            if (layers[l - 1].layer_type == LayerType.Pool2D)
                                total_no_weights += layers[l - 1].conv2DLayer.no_filters * layers[l].conv2DLayer.filterSize[0] * layers[l].conv2DLayer.filterSize[1] * layers[l].conv2DLayer.no_filters;
                            else
                                total_no_weights += layers[l].no_weights;
                            total_no_biases += layers[l].conv2DLayer.no_filters;
                            connections_no_nodes += layers[l].layerSize2D[0] * layers[l].layerSize2D[1] * layers[l].conv2DLayer.N_connections_per_node;
                            break;
                        }
                    case (LayerType.Convolutional3D):
                        {
                            total_no_weights += layers[l].no_weights;
                            total_no_biases += layers[l].conv3DLayer.no_filters;
                            connections_no_nodes += layers[l].layerSize3D[0] * layers[l].layerSize3D[1] * layers[l].layerSize3D[2] * layers[l].conv3DLayer.N_connections_per_node;
                            break;
                        }
                    case (LayerType.Pool2D):
                        {
                            connections_no_nodes += layers[l].layerSize * layers[l - 1].layerSize;
                            break;
                        }
                    case (LayerType.TranspConvolutional2D):
                        {

                            if (Noedify_Utils.Is1DLayerType(layers[l - 1]))
                            {

                            }
                            else
                            {
                                total_no_weights += layers[l].no_weights;
                                //connections_no_nodes += layers[l - 1].layerSize2D[0] * layers[l - 1].layerSize2D[1] * layers[l].conv2DLayer.N_connections_per_node;
                                connections_no_nodes += layers[l].layerSize2D[0] * layers[l].layerSize2D[1] * layers[l].conv2DLayer.N_connections_per_node;
                            }
                            total_no_biases += layers[l].conv2DLayer.no_filters;

                            break;
                        }
                    case (LayerType.TranspConvolutional3D):
                        {

                            if (Noedify_Utils.Is1DLayerType(layers[l - 1]))
                            {

                            }
                            else
                            {
                                total_no_weights += layers[l].no_weights;
                                //connections_no_nodes += layers[l - 1].layerSize2D[0] * layers[l - 1].layerSize2D[1] * layers[l].conv2DLayer.N_connections_per_node;
                                connections_no_nodes += layers[l].layerSize3D[0] * layers[l].layerSize3D[1] * layers[l].layerSize3D[2] * layers[l].conv3DLayer.N_connections_per_node;
                            }
                            total_no_biases += layers[l].conv3DLayer.no_filters;

                            break;
                        }
                    case (LayerType.BatchNorm2D):
                        {
                            total_no_weights += layers[l].no_weights;
                            total_no_biases += layers[l].no_biases;
                            break;
                        }
                    case (LayerType.BatchNorm3D):
                        {
                            total_no_weights += layers[l].no_weights;
                            total_no_biases += layers[l].no_biases;
                            break;
                        }
                    default:
                        {
                            total_no_weights += layers[l - 1].layerSize * layers[l].layerSize;
                            total_no_biases += layers[l].layerSize;
                            break;
                        }
                }

                activeNodeIdx_start_temp.Add(total_no_activeNodes);
                total_no_activeNodes += layers[l].layerSize;
            }
            for (int l = 0; l < LayerCount(); l++)
            {
                nodeIdx_start_temp.Add(total_no_nodes);
                total_no_nodes += layers[l].layerSize;
            }

            networkWeights_par = new NativeArray<float>(total_no_weights, Allocator.Persistent);
            networkBiases_par = new NativeArray<float>(total_no_biases, Allocator.Persistent);
            biasMask_par = new NativeArray<float>(total_no_activeNodes, Allocator.Persistent);
            connections_par = new NativeArray<int>(connections_no_nodes, Allocator.Persistent);
            connectionsInFilter_par = new NativeArray<int>(connections_no_nodes, Allocator.Persistent);
            filterTrack_par = new NativeArray<int>(total_no_activeNodes, Allocator.Persistent);
            nodeTrack_par = new NativeArray<int>(total_no_activeNodes, Allocator.Persistent);

            weightIdx_start = new NativeArray<int>(weightIdx_start_temp.Count, Allocator.Persistent);
            for (int i = 0; i < weightIdx_start_temp.Count; i++) weightIdx_start[i] = weightIdx_start_temp[i];
            biasIdx_start = new NativeArray<int>(biasIdx_start_temp.Count, Allocator.Persistent);
            for (int i = 0; i < biasIdx_start_temp.Count; i++) biasIdx_start[i] = biasIdx_start_temp[i];

            activeNodeIdx_start = new NativeArray<int>(activeNodeIdx_start_temp.Count, Allocator.Persistent);
            for (int i = 0; i < activeNodeIdx_start_temp.Count; i++) activeNodeIdx_start[i] = activeNodeIdx_start_temp[i];
            nodeIdx_start = new NativeArray<int>(nodeIdx_start_temp.Count, Allocator.Persistent);
            for (int i = 0; i < nodeIdx_start_temp.Count; i++) nodeIdx_start[i] = nodeIdx_start_temp[i];
            connectionsIdx_start = new NativeArray<int>(connectionsIdx_start_temp.Count, Allocator.Persistent);
            for (int i = 0; i < connectionsIdx_start_temp.Count; i++) connectionsIdx_start[i] = connectionsIdx_start_temp[i];

            for (int l = 1; l < LayerCount(); l++)
            {
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();
                switch (layers[l].layer_type)
                {
                    case (LayerType.Convolutional2D):
                        {

                            for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                            {
                                for (int j = 0; j < layers[l].conv2DLayer.N_weights_per_filter; j++)
                                {

                                    networkWeights_par[weightIdx_start[l - 1] + f * layers[l].conv2DLayer.N_weights_per_filter + j] = layers[l].weights.valuesConv[f, j];
                                }
                            }
                            for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + f] = layers[l].biases.valuesConv[f];
                            }
                            for (int i = 0; i < layers[l].layerSize2D[0] * layers[l].layerSize2D[1]; i++)
                                for (int j = 0; j < layers[l].conv2DLayer.N_connections_per_node; j++)
                                {
                                    connections_par[connectionsIdx_start[l - 1] + i * layers[l].conv2DLayer.N_connections_per_node + j] = layers[l].conv2DLayer.connections[i, j];
                                    connectionsInFilter_par[connectionsIdx_start[l - 1] + i * layers[l].conv2DLayer.N_connections_per_node + j] = layers[l].conv2DLayer.connectionsInFilter[i, j];
                                }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                filterTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv2DLayer.filterTrack[j];
                                nodeTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv2DLayer.nodeTrack[j];
                                biasMask_par[activeNodeIdx_start[l - 1] + j] = layers[l].biases.valuesConv[layers[l].conv2DLayer.filterTrack[j]];
                            }
                            break;
                        }
                    case (LayerType.Convolutional3D):
                        {

                            for (int f = 0; f < layers[l].conv3DLayer.no_filters; f++)
                            {
                                for (int j = 0; j < layers[l].conv3DLayer.N_weights_per_filter; j++)
                                {

                                    networkWeights_par[weightIdx_start[l - 1] + f * layers[l].conv3DLayer.N_weights_per_filter + j] = layers[l].weights.valuesConv[f, j];
                                }
                            }
                            for (int f = 0; f < layers[l].conv3DLayer.no_filters; f++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + f] = layers[l].biases.valuesConv[f];
                            }
                            for (int i = 0; i < layers[l].layerSize3D[0] * layers[l].layerSize3D[1] * layers[l].layerSize3D[2]; i++)
                                for (int j = 0; j < layers[l].conv3DLayer.N_connections_per_node; j++)
                                {
                                    connections_par[connectionsIdx_start[l - 1] + i * layers[l].conv3DLayer.N_connections_per_node + j] = layers[l].conv3DLayer.connections[i, j];
                                    connectionsInFilter_par[connectionsIdx_start[l - 1] + i * layers[l].conv3DLayer.N_connections_per_node + j] = layers[l].conv3DLayer.connectionsInFilter[i, j];
                                }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                filterTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv3DLayer.filterTrack[j];
                                nodeTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv3DLayer.nodeTrack[j];
                                biasMask_par[activeNodeIdx_start[l - 1] + j] = layers[l].biases.valuesConv[layers[l].conv3DLayer.filterTrack[j]];
                            }
                            break;
                        }
                    case (LayerType.Pool2D):
                        {
                            for (int i = 0; i < layers[l - 1].layerSize; i++)
                                for (int j = 0; j < layers[l].layerSize; j++)
                                {
                                    connectionMask_par[connectionMaskIndeces_start[l - 1] + i * layers[l].layerSize + j] = layers[l].conv2DLayer.connectionMask[i, j];
                                }
                            break;
                        }
                    case (LayerType.TranspConvolutional2D):
                        {
                            if (Noedify_Utils.Is1DLayerType(layers[l - 1]))
                            {

                            }
                            else
                            {
                                int filterSize = layers[l].conv2DLayer.filterSize[0] * layers[l].conv2DLayer.filterSize[1];
                                for (int fj = 0; fj < layers[l].conv2DLayer.no_filters; fj++)
                                {
                                    for (int fi = 0; fi < layers[l - 1].conv2DLayer.no_filters; fi++)
                                    {
                                        for (int conn = 0; conn < filterSize; conn++)
                                        {
                                            networkWeights_par[weightIdx_start[l - 1] + fj * layers[l - 1].conv2DLayer.no_filters * filterSize + fi * filterSize + conn] = layers[l].weights.valuesConv[fj, fi * filterSize + conn];
                                        }
                                    }
                                }
                            }
                            for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + f] = layers[l].biases.valuesConv[f];
                            }
                            for (int j = 0; j < layers[l].layerSize2D[0] * layers[l].layerSize2D[1]; j++)
                                for (int conn = 0; conn < layers[l].conv2DLayer.N_connections_per_node; conn++)
                                {
                                    connections_par[connectionsIdx_start[l - 1] + j * layers[l].conv2DLayer.N_connections_per_node + conn] = layers[l].conv2DLayer.connections[j, conn];
                                    connectionsInFilter_par[connectionsIdx_start[l - 1] + j * layers[l].conv2DLayer.N_connections_per_node + conn] = layers[l].conv2DLayer.connectionsInFilter[j, conn];
                                }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                filterTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv2DLayer.filterTrack[j];
                                nodeTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv2DLayer.nodeTrack[j];
                                biasMask_par[activeNodeIdx_start[l - 1] + j] = layers[l].biases.valuesConv[layers[l].conv2DLayer.filterTrack[j]];
                            }
                            break;
                        }
                    case (LayerType.TranspConvolutional3D):
                        {
                            if (Noedify_Utils.Is1DLayerType(layers[l - 1]))
                            {

                            }
                            else
                            {
                                int filterSize = layers[l].conv3DLayer.filterSize[0] * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.filterSize[2];
                                for (int fj = 0; fj < layers[l].conv3DLayer.no_filters; fj++)
                                {
                                    for (int fi = 0; fi < layers[l - 1].conv3DLayer.no_filters; fi++)
                                    {
                                        for (int conn = 0; conn < filterSize; conn++)
                                        {
                                            networkWeights_par[weightIdx_start[l - 1] + fj * layers[l - 1].conv3DLayer.no_filters * filterSize + fi * filterSize + conn] = layers[l].weights.valuesConv[fj, fi * filterSize + conn];
                                        }
                                    }
                                }
                            }
                            for (int f = 0; f < layers[l].conv3DLayer.no_filters; f++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + f] = layers[l].biases.valuesConv[f];
                            }
                            for (int j = 0; j < layers[l].layerSize3D[0] * layers[l].layerSize3D[1] * layers[l].layerSize3D[2]; j++)
                                for (int conn = 0; conn < layers[l].conv3DLayer.N_connections_per_node; conn++)
                                {
                                    connections_par[connectionsIdx_start[l - 1] + j * layers[l].conv3DLayer.N_connections_per_node + conn] = layers[l].conv3DLayer.connections[j, conn];
                                    connectionsInFilter_par[connectionsIdx_start[l - 1] + j * layers[l].conv3DLayer.N_connections_per_node + conn] = layers[l].conv3DLayer.connectionsInFilter[j, conn];
                                }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                filterTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv3DLayer.filterTrack[j];
                                nodeTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv3DLayer.nodeTrack[j];
                                biasMask_par[activeNodeIdx_start[l - 1] + j] = layers[l].biases.valuesConv[layers[l].conv3DLayer.filterTrack[j]];
                            }
                            break;
                        }
                    case (LayerType.BatchNorm2D):
                        {
                            for (int j = 0; j < layers[l].no_weights; j++)
                            {
                                networkWeights_par[weightIdx_start[l - 1] + j] = layers[l].weights.values[0, j];
                            }
                            for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + f] = layers[l].biases.values[f];
                            }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                filterTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv2DLayer.filterTrack[j];
                                nodeTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv2DLayer.nodeTrack[j];
                                biasMask_par[activeNodeIdx_start[l - 1] + j] = layers[l].biases.values[layers[l].conv2DLayer.filterTrack[j]];
                            }
                            break;
                        }
                    case (LayerType.BatchNorm3D):
                        {
                            for (int j = 0; j < layers[l].no_weights; j++)
                            {
                                networkWeights_par[weightIdx_start[l - 1] + j] = layers[l].weights.values[0, j];
                            }
                            for (int f = 0; f < layers[l].conv3DLayer.no_filters; f++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + f] = layers[l].biases.values[f];
                            }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                filterTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv3DLayer.filterTrack[j];
                                nodeTrack_par[activeNodeIdx_start[l - 1] + j] = layers[l].conv3DLayer.nodeTrack[j];
                                biasMask_par[activeNodeIdx_start[l - 1] + j] = layers[l].biases.values[layers[l].conv3DLayer.filterTrack[j]];
                            }
                            break;
                        }
                    default:
                        {
                            for (int i = 0; i < layers[l - 1].layerSize; i++)
                            {
                                for (int j = 0; j < layers[l].layerSize; j++)
                                {
                                    networkWeights_par[weightIdx_start[l - 1] + i * layers[l].layerSize + j] = layers[l].weights.values[i, j];
                                }
                            }
                            for (int j = 0; j < layers[l].layerSize; j++)
                            {
                                networkBiases_par[biasIdx_start[l - 1] + j] = layers[l].biases.values[j];
                            }
                            break;
                        }
                }
                sw.Stop();
                print("***layer " + l + " nativeparam initialization time: " + sw.ElapsedMilliseconds + " ms");
            }
            nativeArraysInitialized = true;

        }

        public void GenerateComputeBuffers()
        {
            networkWeights_cbuf = new ComputeBuffer(networkWeights_par.Length, sizeof(float));
            networkBiases_cbuf = new ComputeBuffer(networkBiases_par.Length, sizeof(float));
            biasMask_cbuf = new ComputeBuffer(biasMask_par.Length, sizeof(float));
            weightIdx_start_cbuf = new ComputeBuffer(weightIdx_start.Length, sizeof(int));
            biasIdx_start_cbuf = new ComputeBuffer(biasIdx_start.Length, sizeof(int));
            activeNodeIdx_start_cbuf = new ComputeBuffer(activeNodeIdx_start.Length, sizeof(int));
            nodeIdx_start_l0_cbuf = new ComputeBuffer(nodeIdx_start.Length, sizeof(int));
            connections_cbuf = new ComputeBuffer(connections_par.Length, sizeof(int));
            connectionsInFilter_cbuf = new ComputeBuffer(connectionsInFilter_par.Length, sizeof(int));
            filterTrack_cbuf = new ComputeBuffer(filterTrack_par.Length, sizeof(int));
            nodeTrack_cbuf = new ComputeBuffer(nodeTrack_par.Length, sizeof(int));
            connectionsIdx_start_cbuf = new ComputeBuffer(connectionsIdx_start.Length, sizeof(int));

            networkWeights_cbuf.SetData(networkWeights_par);
            networkBiases_cbuf.SetData(networkBiases_par);
            biasMask_cbuf.SetData(biasMask_par);
            weightIdx_start_cbuf.SetData(weightIdx_start);
            biasIdx_start_cbuf.SetData(biasIdx_start);
            activeNodeIdx_start_cbuf.SetData(activeNodeIdx_start);
            nodeIdx_start_l0_cbuf.SetData(nodeIdx_start);
            connections_cbuf.SetData(connections_par);
            connectionsInFilter_cbuf.SetData(connectionsInFilter_par);
            filterTrack_cbuf.SetData(filterTrack_par);
            nodeTrack_cbuf.SetData(nodeTrack_par);
            connectionsIdx_start_cbuf.SetData(connectionsIdx_start);

            cBuffersInitialized = true;

        }

        public void OffloadNativeParameterArrays()
        {
            total_no_nodes = 0;
            total_no_activeNodes = 0;
            total_no_weights = 0;
            total_no_biases = 0;

            // Following arrays have size [LayerCount()-1]
            List<int> weightIndeces_start_temp = new List<int>();
            List<int> biasIndeces_start_temp = new List<int>();
            List<int> activeNodeIndeces_start_temp = new List<int>();
            // Following arrays have size [LayerCount()]
            List<int> nodeIndeces_start_temp = new List<int>();

            for (int l = 1; l < LayerCount(); l++)
            {
                if (layers[l].layer_type == Noedify.LayerType.Convolutional2D)
                {
                    for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                    {
                        for (int j = 0; j < layers[l].conv2DLayer.N_weights_per_filter; j++)
                            layers[l].weights.valuesConv[f, j] = networkWeights_par[weightIdx_start[l - 1] + f * layers[l].conv2DLayer.N_weights_per_filter + j];
                    }
                    for (int f = 0; f < layers[l].conv2DLayer.no_filters; f++)
                    {
                        layers[l].biases.valuesConv[f] = networkBiases_par[biasIdx_start[l - 1] + f];
                    }
                }
                else if (layers[l].layer_type == LayerType.Pool2D)
                {

                }
                else
                {
                    for (int i = 0; i < layers[l - 1].layerSize; i++)
                    {
                        for (int j = 0; j < layers[l].layerSize; j++)
                            layers[l].weights.values[i, j] = networkWeights_par[weightIdx_start[l - 1] + i * layers[l].layerSize + j];
                    }
                    for (int j = 0; j < layers[l].layerSize; j++)
                    {
                        layers[l].biases.values[j] = networkBiases_par[biasIdx_start[l - 1] + j];
                    }
                }
            }
        }

        public void Cleanup_Par()
        {
            CleanupNAifValid(networkWeights_par);
            CleanupNAifValid(networkBiases_par);
            if (networkWeights_gradients_par.IsCreated)
            {
                networkWeights_gradients_par.Dispose();
                networkBiases_gradients_par.Dispose();
                weightGradientIndeces_start.Dispose();
                biasGradientIndeces_start.Dispose();
            }
            CleanupNAifValid(biasMask_par);
            if (connectionMask_par.IsCreated)
            {
                connectionMask_par.Dispose();
                connectionMaskIndeces_start.Dispose();
            }

            CleanupNAifValid(connections_par);
            CleanupNAifValid(connectionsInFilter_par);
            CleanupNAifValid(connectionsIdx_start);
            CleanupNAifValid(weightIdx_start);
            CleanupNAifValid(biasIdx_start);

            CleanupNAifValid(activeNodeIdx_start);
            CleanupNAifValid(nodeIdx_start);
            CleanupNAifValid(filterTrack_par);
            CleanupNAifValid(nodeTrack_par);
            nativeArraysInitialized = false;
        }

        public void Cleanup_ComputeBuffers()
        {
            CleanupCBifValid(networkWeights_cbuf);
            CleanupCBifValid(networkBiases_cbuf);
            CleanupCBifValid(biasMask_cbuf);
            CleanupCBifValid(weightIdx_start_cbuf);
            CleanupCBifValid(biasIdx_start_cbuf);
            CleanupCBifValid(activeNodeIdx_start_cbuf);
            CleanupCBifValid(nodeIdx_start_l0_cbuf);
            CleanupCBifValid(connections_cbuf);
            CleanupCBifValid(connectionsInFilter_cbuf);
            CleanupCBifValid(filterTrack_cbuf);
            CleanupCBifValid(nodeTrack_cbuf);
            CleanupCBifValid(connectionsIdx_start_cbuf);

            cBuffersInitialized = false;
        }

        private static void CleanupNAifValid(NativeArray<float> array)
        {
            if (array.IsCreated)
            {
                try
                {
                    array.Dispose();
                }
                catch { }
            }
        }
        private static void CleanupNAifValid(NativeArray<int> array)
        {
            if (array.IsCreated)
            {
                try
                {
                    array.Dispose();
                }
                catch { }
            }
        }

        private static void CleanupCBifValid(ComputeBuffer cb)
        {
            try
            {
                cb.Dispose();
            }
            catch { }
        }

        #region Utility Functions

        public int LayerCount()
        {
            return layers.Count;
        }

        public int Get_Total_No_Weights()
        {
            int weightCount = 0;
            for (int l = 1; l < LayerCount(); l++)
            {
                int startWeights = weightCount;
                if (layers[l].layer_type == LayerType.Convolutional2D & layers[l - 1].layer_type == LayerType.Pool2D)
                    weightCount += layers[l - 1].conv2DLayer.no_filters * layers[l].conv2DLayer.filterSize[0] * layers[l].conv2DLayer.filterSize[1] * layers[l].conv2DLayer.no_filters;
                else if (layers[l].layer_type == LayerType.Convolutional2D)
                    weightCount += layers[l].in_channels * layers[l].conv2DLayer.filterSize[0] * layers[l].conv2DLayer.filterSize[1] * layers[l].conv2DLayer.no_filters;
                else if (layers[l].layer_type.Equals(LayerType.Convolutional3D))
                    weightCount += layers[l].in_channels * layers[l].conv3DLayer.filterSize[0] * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.no_filters;
                else if ((layers[l].layer_type == LayerType.TranspConvolutional2D) & Noedify_Utils.Is1DLayerType(layers[l - 1])) // TransposeConv2D with previous layer 1D
                    weightCount += layers[l - 1].layerSize * layers[l].conv2DLayer.filterSize[0] * layers[l].conv2DLayer.filterSize[1] * layers[l].in_channels;
                else if ((layers[l].layer_type == LayerType.TranspConvolutional2D)) // TransposeConv2D with previous layer 2D
                    weightCount += layers[l].in_channels * layers[l].conv2DLayer.filterSize[0] * layers[l].conv2DLayer.filterSize[1] * layers[l].conv2DLayer.no_filters;
                else if ((layers[l].layer_type.Equals(LayerType.TranspConvolutional3D)) & Noedify_Utils.Is1DLayerType(layers[l - 1])) // TransposeConv3D with previous layer 1D
                    weightCount += layers[l - 1].layerSize * layers[l].conv3DLayer.filterSize[0] * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.filterSize[2] * layers[l].in_channels;
                else if ((layers[l].layer_type.Equals(LayerType.TranspConvolutional3D))) // TransposeConv3D with previous layer 3D
                    weightCount += layers[l].in_channels * layers[l].conv3DLayer.filterSize[0] * layers[l].conv3DLayer.filterSize[1] * layers[l].conv3DLayer.filterSize[2] *  layers[l].conv3DLayer.no_filters;
                else if (layers[l].layer_type == LayerType.Pool2D) { }
                else if (layers[l].layer_type.Equals(LayerType.BatchNorm2D))
                    weightCount += layers[l].in_channels;
                else if (layers[l].layer_type.Equals(LayerType.BatchNorm3D))
                    weightCount += layers[l].in_channels;
                else
                    weightCount += layers[l].layerSize * layers[l - 1].layerSize;
                //print("layer " + l + " (" + layers[l].name + ") weights: " + (weightCount - startWeights) + "(" + weightCount + ")");

            }
            return weightCount;
        }

        public int Get_Total_No_Biases()
        {
            int biasCount = 0;
            for (int l = 1; l < LayerCount(); l++)
            {
                int startBiasCount = biasCount;
                if (layers[l].layer_type == LayerType.Convolutional2D)
                    biasCount += layers[l].conv2DLayer.no_filters;
                if (layers[l].layer_type.Equals(LayerType.Convolutional3D))
                    biasCount += layers[l].conv3DLayer.no_filters;
                else if (layers[l].layer_type == LayerType.TranspConvolutional2D)
                    biasCount += layers[l].conv2DLayer.no_filters;
                else if (layers[l].layer_type.Equals(LayerType.TranspConvolutional3D))
                    biasCount += layers[l].conv3DLayer.no_filters;
                else if (layers[l].layer_type.Equals(LayerType.BatchNorm2D))
                    biasCount += layers[l].in_channels;
                else if (layers[l].layer_type.Equals(LayerType.BatchNorm3D))
                    biasCount += layers[l].in_channels;
                else
                    biasCount += layers[l].layerSize;
                //print("layer " + l + " (" + layers[l].name + ") biases: " + (biasCount - startBiasCount) + "(" + biasCount + ")");

            }
            return biasCount;
        }

        #endregion
    }

    public static void PrintArrayLine(string name, float[] array, bool includeIndex = false)
    {
        print(name);
        string outString = "";
        for (int i = 0; i < array.Length; i++)
        {
            if (includeIndex)
                outString += "(" + i + ")";
            outString += array[i] + ", ";
        }
        print(outString);
    }
    public static void PrintArrayLine(string name, NativeArray<float> array, int[] bounds, bool includeIndex = false)
    {
        string outString = name + ": ";
        for (int i = bounds[0]; i < bounds[1]; i++)
        {
            if (includeIndex)
                outString += "(" + (i - bounds[0]) + ")";
            outString += array[i] + ", ";
        }
        print(outString);
    }

    public static void PrintTrainingSet(List<float[,,]> trainingSet, int size1D)
    {
        for (int set = 0; set < trainingSet.Count; set++)
        {
            string outString = "Training Input " + set + ": ";

            for (int i = 0; i < size1D; i++)
            {
                outString += trainingSet[set][0, 0, i] + " ";
            }
            print(outString);
        }
    }

    public static void PrintTrainingSet(List<float[]> trainingSet)
    {
        for (int set = 0; set < trainingSet.Count; set++)
        {
            string outString = "Training Output " + set + ": ";

            for (int i = 0; i < trainingSet[set].Length; i++)
            {
                outString += trainingSet[set][i] + " ";
            }
            print(outString);
        }
    }

    static public Noedify_Solver CreateSolver()
    {
        GameObject trainObj = new GameObject("Noedify Solver");
        Noedify_Solver solver = trainObj.AddComponent<Noedify_Solver>();
        solver.evaluationInProgress = false;
        solver.trainingInProgress = false;
        return solver;
    }

    static public void DestroySolver(Noedify_Solver solver)
    {
        Destroy(solver.gameObject);
    }

}
