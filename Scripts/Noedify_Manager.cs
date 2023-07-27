#define NOEDIFY_NORELEASE
//#define CONNMASK
#define COMPRESSCONNECTIONS
#define USECOMPILEABLEPARAMS

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;

public class Noedify_Manager
{
    [System.Serializable]
    public class SingleByteArray
    {
        public byte[] byte0;
        public byte[] byte1;
        public byte[] byte2;
        public byte[] byte3;

        public SingleByteArray(int size)
        {
            byte0 = new byte[size];
            byte1 = new byte[size];
            byte2 = new byte[size];
            byte3 = new byte[size];
        }

        public SingleByteArray()
        {
            byte0 = new byte[1];
            byte1 = new byte[1];
            byte2 = new byte[1];
            byte3 = new byte[1];
        }
    }

    [System.Serializable]
    public class SerializedModel
    {
        public List<Noedify.Layer> layers;
        public float trainingRate;
        public Noedify_Solver.CostFunction costFunction;
        public int total_no_nodes;
        public int total_no_activeNodes;
        public int total_no_weights;
        public int total_no_biases;

        public SerializedModel(Noedify.Net model)
        {
            if (model != null)
            {
                layers = model.layers;
                total_no_nodes = model.total_no_nodes;
                total_no_activeNodes = model.total_no_activeNodes;
                total_no_weights = model.total_no_weights;
                total_no_biases = model.total_no_biases;
            }
        }

        public Noedify.Net ReturnNet()
        {
            Noedify.Net outputModel = new Noedify.Net();
            outputModel.layers = layers;
            outputModel.total_no_nodes = total_no_nodes;
            outputModel.total_no_activeNodes = total_no_activeNodes;
            outputModel.total_no_weights = total_no_weights;
            outputModel.total_no_biases = total_no_biases;
            return outputModel;
        }
    }

    [System.Serializable]
    public class CompressedModel
    {
        public int layerCount;

        public int total_no_nodes;
        public int total_no_activeNodes;
        public int total_no_weights;
        public int total_no_biases;

        List<int> layerSizeList;
        List<int> layer_no;
        List<SingleByteArray> weightValuesList;
        List<SingleByteArray> biasValuesList;
        List<int> noWeightsList;
        List<int> noBiasesList;
        List<Noedify.LayerType> layerTypeList;
        List<int> channelsList;
        List<string> layerNames;
        List<Noedify.ActivationFunction> activationFunctionList;
        List<float> epsList;

        List<int[]> layerSize2DList;
        List<int[]> layerSize3DList;
        List<int[]> filterSizeList;
        List<int> N_connections_per_nodeList;
        List<int> N_weights_per_filterList;
        List<SingleByteArray> connectionsList;
        List<SingleByteArray> connectionsInFilterList;
#if CONNMASK
        List<SingleByteArray> connectionMaskList;
#endif
        List<byte[]> filterTrackList;
        List<byte[]> channelTrackList;
        List<SingleByteArray> nodeTrackList;
        List<int> noFiltersList;

        // Parallel Processing Arrays
        public bool parVariablesSaved;
        SingleByteArray networkWeights_par;
        SingleByteArray networkBiases_par;
        SingleByteArray biasMask_par;
        SingleByteArray weightIdx_start;
        SingleByteArray biasIdx_start;
        SingleByteArray activeNodeIdx_start;
        SingleByteArray nodeIdx_start;
        SingleByteArray connections_par;
        SingleByteArray connectionsInFilter_par;
        SingleByteArray connectionsIdx_start;
        SingleByteArray filterTrack_par;
        SingleByteArray nodeTrack_par;
        int activeNodesSize;
        int nodesSize;
        int[] weightsParSize;
        int[] biasParSize;
        int biasMaskSize;
        int[] connectionsSize;
        int filterTrackSize;
        int nodeTrackSize;

        public void Compress(Noedify.Net model, bool storeParVariables = false)
        {

            layerCount = model.LayerCount();

            total_no_nodes = model.total_no_nodes;
            total_no_activeNodes = model.total_no_activeNodes;
            total_no_weights = model.total_no_weights;
            total_no_biases = model.total_no_biases;

            layerSizeList = new List<int>();
            layer_no = new List<int>();
            weightValuesList = new List<SingleByteArray>();
            biasValuesList = new List<SingleByteArray>();
            noWeightsList = new List<int>();
            noBiasesList = new List<int>();
            layerTypeList = new List<Noedify.LayerType>();
            channelsList = new List<int>();
            layerNames = new List<string>();
            activationFunctionList = new List<Noedify.ActivationFunction>();
            epsList = new List<float>();

            layerSize2DList = new List<int[]>();
            layerSize3DList = new List<int[]>();
            filterSizeList = new List<int[]>();
            N_connections_per_nodeList = new List<int>();
            N_weights_per_filterList = new List<int>();
            connectionsList = new List<SingleByteArray>();
            connectionsInFilterList = new List<SingleByteArray>();
#if CONNMASK
            connectionMaskList = new List<SingleByteArray>();
#endif
            filterTrackList = new List<byte[]>();
            channelTrackList = new List<byte[]>();
            nodeTrackList = new List<SingleByteArray>();
            noFiltersList = new List<int>();

            if (model != null)
            {
                // Compress and store native parameter arrays
                if (model.nativeArraysInitialized & storeParVariables)
                {
                    networkWeights_par = CompressFloatArray(model.networkWeights_par.ToArray(), model.networkWeights_par.Length);
                    networkBiases_par = CompressFloatArray(model.networkBiases_par.ToArray(), model.networkBiases_par.Length);
                    biasMask_par = CompressFloatArray(model.biasMask_par.ToArray(), model.biasMask_par.Length);
                    weightIdx_start = CompressIntArray(model.weightIdx_start.ToArray(), model.weightIdx_start.Length);
                    biasIdx_start = CompressIntArray(model.biasIdx_start.ToArray(), model.biasIdx_start.Length);
                    activeNodeIdx_start = CompressIntArray(model.activeNodeIdx_start.ToArray(), model.activeNodeIdx_start.Length);
                    nodeIdx_start = CompressIntArray(model.nodeIdx_start.ToArray(), model.nodeIdx_start.Length);
                    connections_par = CompressIntArray(model.connections_par.ToArray(), model.connections_par.Length);
                    connectionsInFilter_par = CompressIntArray(model.connectionsInFilter_par.ToArray(), model.connectionsInFilter_par.Length);
                    connectionsIdx_start = CompressIntArray(model.connectionsIdx_start.ToArray(), model.connectionsIdx_start.Length);
                    filterTrack_par = CompressIntArray(model.filterTrack_par.ToArray(), model.filterTrack_par.Length);
                    nodeTrack_par = CompressIntArray(model.nodeTrack_par.ToArray(), model.nodeTrack_par.Length);
                    activeNodesSize = model.activeNodeIdx_start.Length;
                    nodesSize = model.nodeIdx_start.Length;
                    weightsParSize = new int[2] { model.networkWeights_par.Length, model.weightIdx_start.Length };
                    biasParSize = new int[2] { model.networkBiases_par.Length, model.biasIdx_start.Length };
                    biasMaskSize = model.biasMask_par.Length;
                    connectionsSize = new int[2] { model.connections_par.Length, model.connectionsIdx_start.Length };
                    filterTrackSize = model.filterTrack_par.Length;
                    nodeTrackSize = model.nodeTrack_par.Length;
                    parVariablesSaved = true;
                }
                for (int l = 0; l < model.LayerCount(); l++)
                {
                    //Debug.Log("Compressing layer " + l);
                    layer_no.Add(l);
                    layerSizeList.Add(model.layers[l].layerSize);
                    layerSize2DList.Add(model.layers[l].layerSize2D);
                    layerSize3DList.Add(model.layers[l].layerSize3D);
                    layerTypeList.Add(model.layers[l].layer_type);
                    channelsList.Add(model.layers[l].in_channels);
                    layerNames.Add(model.layers[l].name);
                    activationFunctionList.Add(model.layers[l].activationFunction);
                    epsList.Add(model.layers[l].bn_eps);

                    switch (model.layers[l].layer_type)
                    {
                        case (Noedify.LayerType.Input):
                            {
                                weightValuesList.Add(new SingleByteArray());
                                biasValuesList.Add(new SingleByteArray());
                                noWeightsList.Add(0);
                                noBiasesList.Add(0);
                                AddNull1DLayer();
                                break;
                            }
                        case (Noedify.LayerType.Input2D):
                            {
                                weightValuesList.Add(new SingleByteArray());
                                biasValuesList.Add(new SingleByteArray());
                                noWeightsList.Add(0);
                                noBiasesList.Add(0);
                                filterSizeList.Add(new int[2]);
                                connectionsList.Add(new SingleByteArray());
                                connectionsInFilterList.Add(new SingleByteArray());
#if CONNMASK
            connectionMaskList.Add(new SingleByteArray());
#endif
                                filterTrackList.Add(new byte[1]);
                                channelTrackList.Add(new byte[1]);
                                nodeTrackList.Add(new SingleByteArray());
                                noFiltersList.Add(model.layers[l].conv2DLayer.no_filters);
                                N_connections_per_nodeList.Add(0);
                                N_weights_per_filterList.Add(0);
                                break;
                            }
                        case (Noedify.LayerType.Input3D):
                            {
                                weightValuesList.Add(new SingleByteArray());
                                biasValuesList.Add(new SingleByteArray());
                                noWeightsList.Add(0);
                                noBiasesList.Add(0);
                                filterSizeList.Add(new int[3]);
                                connectionsList.Add(new SingleByteArray());
                                connectionsInFilterList.Add(new SingleByteArray());
#if CONNMASK
            connectionMaskList.Add(new SingleByteArray());
#endif
                                filterTrackList.Add(new byte[1]);
                                channelTrackList.Add(new byte[1]);
                                nodeTrackList.Add(new SingleByteArray());
                                noFiltersList.Add(model.layers[l].conv3DLayer.no_filters);
                                N_connections_per_nodeList.Add(0);
                                N_weights_per_filterList.Add(0);
                                break;
                            }
                        case (Noedify.LayerType.Output):
                        case (Noedify.LayerType.FullyConnected):
                            {
                                weightValuesList.Add(CompressFloatArray(model.layers[l].weights.values, model.layers[l - 1].layerSize, model.layers[l].layerSize));
                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.values, model.layers[l].layerSize));
                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);
                                AddNull1DLayer();
                                break;
                            }
                        case (Noedify.LayerType.Convolutional2D):
                            {
                                weightValuesList.Add(CompressFloatArray(model.layers[l].weights.valuesConv, model.layers[l].conv2DLayer.no_filters, model.layers[l].in_channels * model.layers[l].conv2DLayer.filterSize[1] * model.layers[l].conv2DLayer.filterSize[0]));
                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.valuesConv, model.layers[l].conv2DLayer.no_filters));
#if CONNMASK
                                    connectionMaskList.Add(new SingleByteArray());
#endif
                                connectionsList.Add(CompressIntArray(model.layers[l].conv2DLayer.connections, model.layers[l].layerSize2D[0] * model.layers[l].layerSize2D[1], model.layers[l].conv2DLayer.N_connections_per_node));
                                connectionsInFilterList.Add(CompressIntArray(model.layers[l].conv2DLayer.connectionsInFilter, model.layers[l].layerSize2D[0] * model.layers[l].layerSize2D[1], model.layers[l].conv2DLayer.N_connections_per_node));

                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);

                                filterTrackList.Add(ConvertToBytes<int>(model.layers[l].conv2DLayer.filterTrack, model.layers[l].layerSize));
                                channelTrackList.Add(ConvertToBytes<int>(model.layers[l].conv2DLayer.channelTrack, model.layers[l].layerSize));
                                nodeTrackList.Add(CompressIntArray(model.layers[l].conv2DLayer.nodeTrack, model.layers[l].layerSize));
                                noFiltersList.Add(model.layers[l].conv2DLayer.no_filters);
                                filterSizeList.Add(model.layers[l].conv2DLayer.filterSize);

                                N_connections_per_nodeList.Add(model.layers[l].conv2DLayer.N_connections_per_node);
                                N_weights_per_filterList.Add(model.layers[l].conv2DLayer.N_weights_per_filter);
                                break;
                            }
                        case (Noedify.LayerType.Convolutional3D):
                            {
                                weightValuesList.Add(CompressFloatArray(model.layers[l].weights.valuesConv, model.layers[l].conv3DLayer.no_filters, model.layers[l].in_channels * model.layers[l].conv3DLayer.filterSize[2] * model.layers[l].conv3DLayer.filterSize[1] * model.layers[l].conv3DLayer.filterSize[0]));
                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.valuesConv, model.layers[l].conv3DLayer.no_filters));

                                connectionsList.Add(CompressIntArray(model.layers[l].conv3DLayer.connections, model.layers[l].layerSize3D[0] * model.layers[l].layerSize3D[1] * model.layers[l].layerSize3D[2], model.layers[l].conv3DLayer.N_connections_per_node));
                                connectionsInFilterList.Add(CompressIntArray(model.layers[l].conv3DLayer.connectionsInFilter, model.layers[l].layerSize3D[0] * model.layers[l].layerSize3D[1] * model.layers[l].layerSize3D[2], model.layers[l].conv3DLayer.N_connections_per_node));

                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);

                                filterTrackList.Add(ConvertToBytes<int>(model.layers[l].conv3DLayer.filterTrack, model.layers[l].layerSize));
                                channelTrackList.Add(ConvertToBytes<int>(model.layers[l].conv3DLayer.channelTrack, model.layers[l].layerSize));
                                nodeTrackList.Add(CompressIntArray(model.layers[l].conv3DLayer.nodeTrack, model.layers[l].layerSize));
                                noFiltersList.Add(model.layers[l].conv3DLayer.no_filters);
                                filterSizeList.Add(model.layers[l].conv3DLayer.filterSize);

                                N_connections_per_nodeList.Add(model.layers[l].conv3DLayer.N_connections_per_node);
                                N_weights_per_filterList.Add(model.layers[l].conv3DLayer.N_weights_per_filter);
                                break;
                            }
                        case (Noedify.LayerType.BatchNorm2D):
                            {
                                weightValuesList.Add(CompressFloatArray(model.layers[l].weights.values, 1, model.layers[l].in_channels));
                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.values, model.layers[l].in_channels));
                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);
                                filterSizeList.Add(model.layers[l].conv2DLayer.filterSize);

                                connectionsList.Add(new SingleByteArray());
                                connectionsInFilterList.Add(new SingleByteArray());
#if CONNMASK
                                connectionMaskList.Add(new SingleByteArray());
#endif
                                filterTrackList.Add(ConvertToBytes<int>(model.layers[l].conv2DLayer.filterTrack, model.layers[l].layerSize));
                                channelTrackList.Add(ConvertToBytes<int>(model.layers[l].conv2DLayer.channelTrack, model.layers[l].layerSize));
                                nodeTrackList.Add(CompressIntArray(model.layers[l].conv2DLayer.nodeTrack, model.layers[l].layerSize));

                                noFiltersList.Add(model.layers[l].conv2DLayer.no_filters);

                                N_connections_per_nodeList.Add(0);
                                N_weights_per_filterList.Add(0);
                                break;
                            }
                        case (Noedify.LayerType.BatchNorm3D):
                            {
                                weightValuesList.Add(CompressFloatArray(model.layers[l].weights.values, 1, model.layers[l].in_channels));
                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.values, model.layers[l].in_channels));
                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);
                                filterSizeList.Add(model.layers[l].conv3DLayer.filterSize);

                                connectionsList.Add(new SingleByteArray());
                                connectionsInFilterList.Add(new SingleByteArray());
#if CONNMASK
                                connectionMaskList.Add(new SingleByteArray());
#endif
                                filterTrackList.Add(ConvertToBytes<int>(model.layers[l].conv3DLayer.filterTrack, model.layers[l].layerSize));
                                channelTrackList.Add(ConvertToBytes<int>(model.layers[l].conv3DLayer.channelTrack, model.layers[l].layerSize));
                                nodeTrackList.Add(CompressIntArray(model.layers[l].conv3DLayer.nodeTrack, model.layers[l].layerSize));

                                noFiltersList.Add(model.layers[l].conv3DLayer.no_filters);

                                N_connections_per_nodeList.Add(0);
                                N_weights_per_filterList.Add(0);
                                break;
                            }
                        case (Noedify.LayerType.TranspConvolutional2D):
                            {
                                if (Noedify_Utils.Is1DLayerType(model.layers[l - 1]))
                                {
                                    weightValuesList.Add(CompressFloatArray(model.layers[l].weights.values, model.layers[l - 1].layerSize, model.layers[l].layerSize));
#if CONNMASK
                                    connectionMaskList.Add(CompressIntArray(model.layers[l].conv2DLayer.connectionMask, model.layers[l - 1].layerSize, model.layers[l].layerSize));
#endif
                                    connectionsList.Add(new SingleByteArray());
                                    connectionsInFilterList.Add(new SingleByteArray());
                                }
                                else
                                {
                                    weightValuesList.Add(CompressFloatArray(model.layers[l].weights.valuesConv, model.layers[l].conv2DLayer.no_filters, model.layers[l - 1].conv2DLayer.no_filters * model.layers[l].conv2DLayer.filterSize[1] * model.layers[l].conv2DLayer.filterSize[0]));
#if CONNMASK
                                    connectionMaskList.Add(new SingleByteArray());
#endif
                                    //                                   connectionsList.Add(CompressIntArray(model.layers[l].conv2DLayer.connections, model.layers[l - 1].layerSize2D[0] * model.layers[l - 1].layerSize2D[1], model.layers[l].conv2DLayer.N_connections_per_node));
                                    //                                   connectionsInFilterList.Add(CompressIntArray(model.layers[l].conv2DLayer.connectionsInFilter, model.layers[l - 1].layerSize2D[0] * model.layers[l - 1].layerSize2D[1], model.layers[l].conv2DLayer.N_connections_per_node));
                                    connectionsList.Add(CompressIntArray(model.layers[l].conv2DLayer.connections, model.layers[l].layerSize2D[0] * model.layers[l].layerSize2D[1], model.layers[l].conv2DLayer.N_connections_per_node));
                                    connectionsInFilterList.Add(CompressIntArray(model.layers[l].conv2DLayer.connectionsInFilter, model.layers[l].layerSize2D[0] * model.layers[l].layerSize2D[1], model.layers[l].conv2DLayer.N_connections_per_node));

                                }

                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.valuesConv, model.layers[l].in_channels));

                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);

                                filterTrackList.Add(ConvertToBytes<int>(model.layers[l].conv2DLayer.filterTrack, model.layers[l].layerSize));
                                channelTrackList.Add(ConvertToBytes<int>(model.layers[l].conv2DLayer.channelTrack, model.layers[l].layerSize));
                                nodeTrackList.Add(CompressIntArray(model.layers[l].conv2DLayer.nodeTrack, model.layers[l].layerSize));
                                noFiltersList.Add(model.layers[l].conv2DLayer.no_filters);
                                filterSizeList.Add(model.layers[l].conv2DLayer.filterSize);

                                N_connections_per_nodeList.Add(model.layers[l].conv2DLayer.N_connections_per_node);
                                N_weights_per_filterList.Add(model.layers[l].conv2DLayer.N_weights_per_filter);

                                break;
                            }
                        case (Noedify.LayerType.TranspConvolutional3D):
                            {
                                if (Noedify_Utils.Is1DLayerType(model.layers[l - 1]))
                                {
                                    weightValuesList.Add(CompressFloatArray(model.layers[l].weights.values, model.layers[l - 1].layerSize, model.layers[l].layerSize));
                                    connectionsList.Add(new SingleByteArray());
                                    connectionsInFilterList.Add(new SingleByteArray());
                                }
                                else
                                {
                                    weightValuesList.Add(CompressFloatArray(model.layers[l].weights.valuesConv, model.layers[l].conv3DLayer.no_filters, model.layers[l - 1].conv3DLayer.no_filters * model.layers[l].conv3DLayer.filterSize[2] * model.layers[l].conv3DLayer.filterSize[1] * model.layers[l].conv3DLayer.filterSize[0]));

                                    connectionsList.Add(CompressIntArray(model.layers[l].conv3DLayer.connections, model.layers[l].layerSize3D[0] * model.layers[l].layerSize3D[1] * model.layers[l].layerSize3D[2], model.layers[l].conv3DLayer.N_connections_per_node));
                                    connectionsInFilterList.Add(CompressIntArray(model.layers[l].conv3DLayer.connectionsInFilter, model.layers[l].layerSize3D[0] * model.layers[l].layerSize3D[1] * model.layers[l].layerSize3D[2], model.layers[l].conv3DLayer.N_connections_per_node));

                                }

                                biasValuesList.Add(CompressFloatArray(model.layers[l].biases.valuesConv, model.layers[l].in_channels));

                                noWeightsList.Add(model.layers[l].no_weights);
                                noBiasesList.Add(model.layers[l].no_biases);

                                filterTrackList.Add(ConvertToBytes<int>(model.layers[l].conv3DLayer.filterTrack, model.layers[l].layerSize));
                                channelTrackList.Add(ConvertToBytes<int>(model.layers[l].conv3DLayer.channelTrack, model.layers[l].layerSize));
                                nodeTrackList.Add(CompressIntArray(model.layers[l].conv3DLayer.nodeTrack, model.layers[l].layerSize));
                                noFiltersList.Add(model.layers[l].conv3DLayer.no_filters);
                                filterSizeList.Add(model.layers[l].conv3DLayer.filterSize);

                                N_connections_per_nodeList.Add(model.layers[l].conv3DLayer.N_connections_per_node);
                                N_weights_per_filterList.Add(model.layers[l].conv3DLayer.N_weights_per_filter);

                                break;
                            }
                        default: Debug.Log("Warning (Noedify_Manager.CompressedModel.Compress()): Unkown layer case " + model.layers[l].layer_type.ToString()); break;
                    }
                }
                for (int l = 0; l < model.LayerCount(); l++)
                {
                    filterTrackList[l] = CompressData(filterTrackList[l]);
                    channelTrackList[l] = CompressData(channelTrackList[l]);
                }

            }
        }
        public void Unwrap(ref Noedify.Net net, bool retrieveParVariables = false)
        {
            net.total_no_nodes = total_no_nodes;
            net.total_no_activeNodes = total_no_activeNodes;
            net.total_no_weights = total_no_weights;
            net.total_no_biases = total_no_biases;

            for (int l = 0; l < layerCount; l++)
            {
                filterTrackList[l] = DecompressData(filterTrackList[l]);
                channelTrackList[l] = DecompressData(channelTrackList[l]);
            }

            net.layers = new List<Noedify.Layer>();

            if (retrieveParVariables & parVariablesSaved)
            {
                net.networkWeights_par = new Unity.Collections.NativeArray<float>(DecompressFloatArray(networkWeights_par, weightsParSize[0]), Unity.Collections.Allocator.Persistent);
                net.weightIdx_start = new Unity.Collections.NativeArray<int>(DecompressIntArray(weightIdx_start, weightsParSize[1]), Unity.Collections.Allocator.Persistent);
                net.networkBiases_par = new Unity.Collections.NativeArray<float>(DecompressFloatArray(networkBiases_par, biasParSize[0]), Unity.Collections.Allocator.Persistent);
                net.biasIdx_start = new Unity.Collections.NativeArray<int>(DecompressIntArray(biasIdx_start, biasParSize[1]), Unity.Collections.Allocator.Persistent);
                net.biasMask_par = new Unity.Collections.NativeArray<float>(DecompressFloatArray(biasMask_par, biasMaskSize), Unity.Collections.Allocator.Persistent);
                net.activeNodeIdx_start = new Unity.Collections.NativeArray<int>(DecompressIntArray(activeNodeIdx_start, activeNodesSize), Unity.Collections.Allocator.Persistent);
                net.nodeIdx_start = new Unity.Collections.NativeArray<int>(DecompressIntArray(nodeIdx_start, nodesSize), Unity.Collections.Allocator.Persistent);
                net.connections_par = new Unity.Collections.NativeArray<int>(DecompressIntArray(connections_par, connectionsSize[0]), Unity.Collections.Allocator.Persistent);
                net.connections_par = new Unity.Collections.NativeArray<int>(DecompressIntArray(connections_par, connectionsSize[0]), Unity.Collections.Allocator.Persistent);
                net.connectionsInFilter_par = new Unity.Collections.NativeArray<int>(DecompressIntArray(connectionsInFilter_par, connectionsSize[0]), Unity.Collections.Allocator.Persistent);
                net.connectionsIdx_start = new Unity.Collections.NativeArray<int>(DecompressIntArray(connectionsIdx_start, connectionsSize[1]), Unity.Collections.Allocator.Persistent);
                net.filterTrack_par = new Unity.Collections.NativeArray<int>(DecompressIntArray(filterTrack_par, filterTrackSize), Unity.Collections.Allocator.Persistent);
                net.nodeTrack_par = new Unity.Collections.NativeArray<int>(DecompressIntArray(nodeTrack_par, nodeTrackSize), Unity.Collections.Allocator.Persistent);
                net.nativeArraysInitialized = true;
            }


            for (int l = 0; l < layerCount; l++)
            {

                //Debug.Log("Unwrapping layer: " + l);
                Noedify.Layer newLayer = new Noedify.Layer(Noedify.LayerType.FullyConnected, 1, layerNames[l]);
                newLayer.layer_no = layer_no[l];
                newLayer.layerSize = layerSizeList[l];
                newLayer.layerSize2D = layerSize2DList[l];
                newLayer.layerSize3D = layerSize3DList[l];
                newLayer.layer_type = layerTypeList[l];
                newLayer.in_channels = channelsList[l];
                newLayer.activationFunction = activationFunctionList[l];
                newLayer.bn_eps = epsList[l];

                newLayer.weights = new Noedify.NN_Weights(1, 1, false);
                newLayer.biases = new Noedify.NN_Biases(1, false);

                switch (newLayer.layer_type)
                {

                    case (Noedify.LayerType.Input):
                        {
                            break;
                        }
                    case (Noedify.LayerType.Input2D):
                        {
                            newLayer.conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                            newLayer.conv2DLayer.no_filters = noFiltersList[l];
                            break;
                        }
                    case (Noedify.LayerType.Input3D):
                        {
                            newLayer.conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                            newLayer.conv3DLayer.no_filters = noFiltersList[l];
                            break;
                        }
                    case (Noedify.LayerType.Output):
                    case (Noedify.LayerType.FullyConnected):
                        {
                            newLayer.weights.values = DecompressFloatArray(weightValuesList[l], net.layers[l - 1].layerSize, newLayer.layerSize);
                            newLayer.biases.values = DecompressFloatArray(biasValuesList[l], newLayer.layerSize);
                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];
                            break;
                        }
                    case (Noedify.LayerType.Convolutional2D):
                        {
                            newLayer.conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                            newLayer.conv2DLayer.no_filters = noFiltersList[l];
                            newLayer.conv2DLayer.N_connections_per_node = N_connections_per_nodeList[l];
                            newLayer.conv2DLayer.N_weights_per_filter = N_weights_per_filterList[l];
                            newLayer.conv2DLayer.filterSize = filterSizeList[l];

                            newLayer.weights.valuesConv = DecompressFloatArray(weightValuesList[l], newLayer.conv2DLayer.no_filters, net.layers[l - 1].conv2DLayer.no_filters * newLayer.conv2DLayer.filterSize[0] * newLayer.conv2DLayer.filterSize[1]);
                            newLayer.biases.valuesConv = DecompressFloatArray(biasValuesList[l], newLayer.conv2DLayer.no_filters);

                            newLayer.conv2DLayer.connections = DecompressIntArray(connectionsList[l], newLayer.layerSize2D[0] * newLayer.layerSize2D[1], newLayer.conv2DLayer.N_connections_per_node);
                            newLayer.conv2DLayer.connectionsInFilter = DecompressIntArray(connectionsInFilterList[l], newLayer.layerSize2D[0] * newLayer.layerSize2D[1], newLayer.conv2DLayer.N_connections_per_node);

                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];
                            newLayer.conv2DLayer.filterTrack = ConvertIntFromBytes(filterTrackList[l], newLayer.layerSize);
                            newLayer.conv2DLayer.channelTrack = ConvertIntFromBytes(channelTrackList[l], newLayer.layerSize);
                            newLayer.conv2DLayer.nodeTrack = DecompressIntArray(nodeTrackList[l], newLayer.layerSize);
                            break;
                        }
                    case (Noedify.LayerType.Convolutional3D):
                        {
                            newLayer.conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                            newLayer.conv3DLayer.no_filters = noFiltersList[l];
                            newLayer.conv3DLayer.N_connections_per_node = N_connections_per_nodeList[l];
                            newLayer.conv3DLayer.N_weights_per_filter = N_weights_per_filterList[l];
                            newLayer.conv3DLayer.filterSize = filterSizeList[l];

                            newLayer.weights.valuesConv = DecompressFloatArray(weightValuesList[l], newLayer.conv3DLayer.no_filters, net.layers[l - 1].conv3DLayer.no_filters * newLayer.conv3DLayer.filterSize[0] * newLayer.conv3DLayer.filterSize[1] * newLayer.conv3DLayer.filterSize[2]);
                            newLayer.biases.valuesConv = DecompressFloatArray(biasValuesList[l], newLayer.conv3DLayer.no_filters);

                            newLayer.conv3DLayer.connections = DecompressIntArray(connectionsList[l], newLayer.layerSize3D[0] * newLayer.layerSize3D[1] * newLayer.layerSize3D[2], newLayer.conv3DLayer.N_connections_per_node);
                            newLayer.conv3DLayer.connectionsInFilter = DecompressIntArray(connectionsInFilterList[l], newLayer.layerSize3D[0] * newLayer.layerSize3D[1] * newLayer.layerSize3D[2], newLayer.conv3DLayer.N_connections_per_node);

                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];
                            newLayer.conv3DLayer.filterTrack = ConvertIntFromBytes(filterTrackList[l], newLayer.layerSize);
                            newLayer.conv3DLayer.channelTrack = ConvertIntFromBytes(channelTrackList[l], newLayer.layerSize);
                            newLayer.conv3DLayer.nodeTrack = DecompressIntArray(nodeTrackList[l], newLayer.layerSize);
                            break;
                        }
                    case (Noedify.LayerType.BatchNorm2D):
                        {
                            newLayer.weights.values = DecompressFloatArray(weightValuesList[l], 1, newLayer.in_channels);
                            newLayer.biases.values = DecompressFloatArray(biasValuesList[l], newLayer.in_channels);

                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];

                            newLayer.conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                            newLayer.conv2DLayer.filterTrack = ConvertIntFromBytes(filterTrackList[l], newLayer.layerSize);
                            newLayer.conv2DLayer.channelTrack = ConvertIntFromBytes(channelTrackList[l], newLayer.layerSize);
                            newLayer.conv2DLayer.nodeTrack = DecompressIntArray(nodeTrackList[l], newLayer.layerSize);

                            newLayer.conv2DLayer.no_filters = noFiltersList[l];
                            newLayer.conv2DLayer.filterSize = filterSizeList[l];

                            break;
                        }
                    case (Noedify.LayerType.BatchNorm3D):
                        {
                            newLayer.weights.values = DecompressFloatArray(weightValuesList[l], 1, newLayer.in_channels);
                            newLayer.biases.values = DecompressFloatArray(biasValuesList[l], newLayer.in_channels);

                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];

                            newLayer.conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                            newLayer.conv3DLayer.filterTrack = ConvertIntFromBytes(filterTrackList[l], newLayer.layerSize);
                            newLayer.conv3DLayer.channelTrack = ConvertIntFromBytes(channelTrackList[l], newLayer.layerSize);
                            newLayer.conv3DLayer.nodeTrack = DecompressIntArray(nodeTrackList[l], newLayer.layerSize);

                            newLayer.conv3DLayer.no_filters = noFiltersList[l];
                            newLayer.conv3DLayer.filterSize = filterSizeList[l];

                            break;
                        }
                    case (Noedify.LayerType.TranspConvolutional2D):
                        {
                            newLayer.conv2DLayer = new Noedify_Convolutional2D.Convolutional2DLayer();
                            newLayer.conv2DLayer.no_filters = noFiltersList[l];
                            newLayer.conv2DLayer.N_connections_per_node = N_connections_per_nodeList[l];
                            newLayer.conv2DLayer.N_weights_per_filter = N_weights_per_filterList[l];
                            newLayer.conv2DLayer.filterSize = filterSizeList[l];
                            if (Noedify_Utils.Is1DLayerType(net.layers[l - 1]))
                            {
                                newLayer.weights.values = DecompressFloatArray(weightValuesList[l], net.layers[l - 1].layerSize, newLayer.layerSize);
#if CONNMASK
                                newLayer.conv2DLayer.connectionMask = DecompressIntArray(connectionMaskList[l], net.layers[l - 1].layerSize, newLayer.layerSize);
#endif
                            }
                            else
                            {
                                newLayer.weights.valuesConv = DecompressFloatArray(weightValuesList[l], newLayer.conv2DLayer.no_filters, net.layers[l - 1].conv2DLayer.no_filters * newLayer.conv2DLayer.filterSize[0] * newLayer.conv2DLayer.filterSize[1]);
                                newLayer.conv2DLayer.connections = DecompressIntArray(connectionsList[l], newLayer.layerSize2D[0] * newLayer.layerSize2D[1], newLayer.conv2DLayer.N_connections_per_node);
                                newLayer.conv2DLayer.connectionsInFilter = DecompressIntArray(connectionsInFilterList[l], newLayer.layerSize2D[0] * newLayer.layerSize2D[1], newLayer.conv2DLayer.N_connections_per_node);
                            }
                            newLayer.biases.valuesConv = DecompressFloatArray(biasValuesList[l], newLayer.in_channels);

                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];

                            newLayer.conv2DLayer.filterTrack = ConvertIntFromBytes(filterTrackList[l], newLayer.layerSize);
                            newLayer.conv2DLayer.channelTrack = ConvertIntFromBytes(channelTrackList[l], newLayer.layerSize);
                            newLayer.conv2DLayer.nodeTrack = DecompressIntArray(nodeTrackList[l], newLayer.layerSize);

                            break;
                        }
                    case (Noedify.LayerType.TranspConvolutional3D):
                        {
                            newLayer.conv3DLayer = new Noedify_Convolutional3D.Convolutional3DLayer();
                            newLayer.conv3DLayer.no_filters = noFiltersList[l];
                            newLayer.conv3DLayer.N_connections_per_node = N_connections_per_nodeList[l];
                            newLayer.conv3DLayer.N_weights_per_filter = N_weights_per_filterList[l];
                            newLayer.conv3DLayer.filterSize = filterSizeList[l];
                            if (Noedify_Utils.Is1DLayerType(net.layers[l - 1]))
                            {
                                newLayer.weights.values = DecompressFloatArray(weightValuesList[l], net.layers[l - 1].layerSize, newLayer.layerSize);
                            }
                            else
                            {
                                newLayer.weights.valuesConv = DecompressFloatArray(weightValuesList[l], newLayer.conv3DLayer.no_filters, net.layers[l - 1].conv3DLayer.no_filters * newLayer.conv3DLayer.filterSize[0] * newLayer.conv3DLayer.filterSize[1] * newLayer.conv3DLayer.filterSize[2]);
                                newLayer.conv3DLayer.connections = DecompressIntArray(connectionsList[l], newLayer.layerSize3D[0] * newLayer.layerSize3D[1] * newLayer.layerSize3D[2], newLayer.conv3DLayer.N_connections_per_node);
                                newLayer.conv3DLayer.connectionsInFilter = DecompressIntArray(connectionsInFilterList[l], newLayer.layerSize3D[0] * newLayer.layerSize3D[1] * newLayer.layerSize3D[2], newLayer.conv3DLayer.N_connections_per_node);
                            }
                            newLayer.biases.valuesConv = DecompressFloatArray(biasValuesList[l], newLayer.in_channels);

                            newLayer.no_weights = noWeightsList[l];
                            newLayer.no_biases = noBiasesList[l];

                            newLayer.conv3DLayer.filterTrack = ConvertIntFromBytes(filterTrackList[l], newLayer.layerSize);
                            newLayer.conv3DLayer.channelTrack = ConvertIntFromBytes(channelTrackList[l], newLayer.layerSize);
                            newLayer.conv3DLayer.nodeTrack = DecompressIntArray(nodeTrackList[l], newLayer.layerSize);

                            break;
                        }
                    default: Debug.Log("Warning (Noedify_Manager.CompressedModel.Unwrap()): Unkown layer case " + newLayer.layer_type.ToString()); break;

                }
                net.AddLayer(newLayer);
            }

        }
        void AddNull1DLayer()
        {
            filterSizeList.Add(new int[2]);
            connectionsList.Add(new SingleByteArray());
            connectionsInFilterList.Add(new SingleByteArray());
#if CONNMASK
            connectionMaskList.Add(new SingleByteArray());
#endif
            filterTrackList.Add(new byte[1]);
            channelTrackList.Add(new byte[1]);
            nodeTrackList.Add(new SingleByteArray());
            noFiltersList.Add(1);
            N_connections_per_nodeList.Add(0);
            N_weights_per_filterList.Add(0);
        }
    }

    static public bool CheckNetworkMatch(Noedify.Net ogNet, Noedify.Net newNet)
    {
        bool match = true;
        int layerCount = ogNet.LayerCount();
        if (layerCount != newNet.LayerCount())
        {
            Debug.Log("CheckNetworkMatch: layer count mistch (orig, new) = (" + layerCount + "," + newNet.LayerCount() + ")");
            match = false;
        }


        for (int l = 0; l < layerCount; l++)
        {

            Noedify.Layer ogLayer = ogNet.layers[l];
            Noedify.Layer newLayer = newNet.layers[l];

            if (ogNet.layers[l].layer_no != newNet.layers[l].layer_no)
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): layer_no mistch(orig, new) = (" + ogNet.layers[l].layer_no + ", " + newNet.layers[l].layer_no + ")");
                match = false;
            }
            if (ogNet.layers[l].layerSize != newNet.layers[l].layerSize)
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): layerSize mistch(orig, new) = (" + ogNet.layers[l].layerSize + ", " + newNet.layers[l].layerSize + ")");
                match = false;
            }
            if (ogLayer.layerSize2D[0] != newLayer.layerSize2D[0] | ogLayer.layerSize2D[1] != newLayer.layerSize2D[1])
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): layerSize2D mistch(orig, new) = ([" + ogNet.layers[l].layerSize2D[0] + "," + ogNet.layers[l].layerSize2D[1] + "],[" + newNet.layers[l].layerSize2D[0] + "," + newNet.layers[l].layerSize2D[1] + "])");
                match = false;
            }
            if (ogLayer.layerSize3D[0] != newLayer.layerSize3D[0] | ogLayer.layerSize3D[1] != newLayer.layerSize3D[1] | ogLayer.layerSize3D[2] != newLayer.layerSize3D[2])
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): layerSize3D mistch(orig, new) = ([" + ogNet.layers[l].layerSize3D[0] + "," + ogNet.layers[l].layerSize3D[1] + "," + ogNet.layers[l].layerSize3D[2] + "],[" + newNet.layers[l].layerSize3D[0] + "," + newNet.layers[l].layerSize3D[1] + "," + newNet.layers[l].layerSize3D[1] + "])");
                match = false;
            }
            if (ogNet.layers[l].layer_type != newNet.layers[l].layer_type)
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): layer_type mistch(orig, new) = (" + ogNet.layers[l].layer_type.ToString() + ", " + newNet.layers[l].layer_type.ToString() + ")");
                match = false;
            }
            if (ogNet.layers[l].in_channels != newNet.layers[l].in_channels)
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): channels mistch(orig, new) = (" + ogNet.layers[l].in_channels + ", " + newNet.layers[l].in_channels + ")");
                match = false;
            }
            if (ogNet.layers[l].activationFunction != newNet.layers[l].activationFunction)
            {
                Debug.Log("CheckNetworkMatch (l=" + l + "): activationFunction mistch(orig, new) = (" + ogNet.layers[l].activationFunction + ", " + newNet.layers[l].activationFunction + ")");
                match = false;
            }

            switch (ogNet.layers[l].layer_type)
            {
                case (Noedify.LayerType.Input):
                case (Noedify.LayerType.Input2D):
                case (Noedify.LayerType.Input3D):
                    {
                        break;
                    }
                case (Noedify.LayerType.Output):
                case (Noedify.LayerType.FullyConnected):
                    {
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.values, newLayer.weights.values))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.values, newLayer.biases.values))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }

                        break;
                    }
                case (Noedify.LayerType.Convolutional2D):
                    {
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.valuesConv, newLayer.weights.valuesConv, "l=" + l + " Convolutional2D weights"))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.connections, newLayer.conv2DLayer.connections, "l = " + l + " Convolutional2D connections"))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsList mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.connectionsInFilter, newLayer.conv2DLayer.connectionsInFilter))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsInFilter mismatch");
                            match = false;
                        }

                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.valuesConv, newLayer.biases.valuesConv))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }

                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.filterTrack, newLayer.conv2DLayer.filterTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  filterTrack mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.channelTrack, newLayer.conv2DLayer.channelTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  channelTrack mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.no_filters != newLayer.conv2DLayer.no_filters)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  no_filters mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.N_connections_per_node != newLayer.conv2DLayer.N_connections_per_node)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_connections_per_node mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.N_weights_per_filter != newLayer.conv2DLayer.N_weights_per_filter)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_weights_per_filter mismatch");
                            match = false;
                        }
                        break;
                    }
                case (Noedify.LayerType.Convolutional3D):
                    {
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.valuesConv, newLayer.weights.valuesConv, "l=" + l + " Convolutional3D weights"))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.connections, newLayer.conv3DLayer.connections, "l = " + l + " Convolutional3D connections"))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsList mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.connectionsInFilter, newLayer.conv3DLayer.connectionsInFilter))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsInFilter mismatch");
                            match = false;
                        }

                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.valuesConv, newLayer.biases.valuesConv))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }

                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.filterTrack, newLayer.conv3DLayer.filterTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  filterTrack mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.channelTrack, newLayer.conv3DLayer.channelTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  channelTrack mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.no_filters != newLayer.conv3DLayer.no_filters)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  no_filters mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.N_connections_per_node != newLayer.conv3DLayer.N_connections_per_node)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_connections_per_node mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.N_weights_per_filter != newLayer.conv3DLayer.N_weights_per_filter)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_weights_per_filter mismatch");
                            match = false;
                        }
                        break;
                    }
                case (Noedify.LayerType.BatchNorm2D):
                    {
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.values, newLayer.weights.values))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.values, newLayer.biases.values))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.filterTrack, newLayer.conv2DLayer.filterTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  filterTrack mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.channelTrack, newLayer.conv2DLayer.channelTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  channelTrack mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.no_filters != newLayer.conv2DLayer.no_filters)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  no_filters mismatch");
                            match = false;
                        }
                        if (ogLayer.bn_eps != newLayer.bn_eps)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  eps mismatch");
                            match = false;
                        }

                        break;

                    }
                case (Noedify.LayerType.BatchNorm3D):
                    {
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.values, newLayer.weights.values))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.values, newLayer.biases.values))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.filterTrack, newLayer.conv3DLayer.filterTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  filterTrack mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.channelTrack, newLayer.conv3DLayer.channelTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  channelTrack mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.no_filters != newLayer.conv3DLayer.no_filters)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  no_filters mismatch");
                            match = false;
                        }
                        if (ogLayer.bn_eps != newLayer.bn_eps)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  eps mismatch");
                            match = false;
                        }

                        break;

                    }
                case (Noedify.LayerType.TranspConvolutional2D):
                    {
                        if (Noedify_Utils.Is1DLayerType(ogNet.layers[l - 1]))
                        {
                            if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.values, newLayer.weights.values, "weight"))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                                match = false;
                            }
                            if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.connectionMask, newLayer.conv2DLayer.connectionMask))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionMask mismatch");
                                match = false;
                            }
                        }
                        else
                        {
                            if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.valuesConv, newLayer.weights.valuesConv, "l=" + l + " TranspConv2D weights"))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                                match = false;
                            }
                            if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.connections, newLayer.conv2DLayer.connections, "l = " + l + " TranspConv2D connections"))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsList mismatch");
                                match = false;
                            }
                            if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.connectionsInFilter, newLayer.conv2DLayer.connectionsInFilter))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsInFilter mismatch");
                                match = false;
                            }
                        }
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.valuesConv, newLayer.biases.valuesConv))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }

                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.filterTrack, newLayer.conv2DLayer.filterTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  filterTrack mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv2DLayer.channelTrack, newLayer.conv2DLayer.channelTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  channelTrack mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.no_filters != newLayer.conv2DLayer.no_filters)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  no_filters mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.N_connections_per_node != newLayer.conv2DLayer.N_connections_per_node)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_connections_per_node mismatch");
                            match = false;
                        }
                        if (ogLayer.conv2DLayer.N_weights_per_filter != newLayer.conv2DLayer.N_weights_per_filter)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_weights_per_filter mismatch");
                            match = false;
                        }
                        break;
                    }
                case (Noedify.LayerType.TranspConvolutional3D):
                    {
                        if (Noedify_Utils.Is1DLayerType(ogNet.layers[l - 1]))
                        {
                            if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.values, newLayer.weights.values, "weight"))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                                match = false;
                            }
                            if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.connectionMask, newLayer.conv3DLayer.connectionMask))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionMask mismatch");
                                match = false;
                            }
                        }
                        else
                        {
                            if (!Noedify_Utils.CompareArray<float>(ogLayer.weights.valuesConv, newLayer.weights.valuesConv, "l=" + l + " TranspConv3D weights"))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  weight mismatch");
                                match = false;
                            }
                            if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.connections, newLayer.conv3DLayer.connections, "l = " + l + " TranspConv3D connections"))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsList mismatch");
                                match = false;
                            }
                            if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.connectionsInFilter, newLayer.conv3DLayer.connectionsInFilter))
                            {
                                Debug.Log("CheckNetworkMatch (l=" + l + "):  connectionsInFilter mismatch");
                                match = false;
                            }
                        }
                        if (!Noedify_Utils.CompareArray<float>(ogLayer.biases.valuesConv, newLayer.biases.valuesConv))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  bias mismatch");
                            match = false;
                        }

                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.filterTrack, newLayer.conv3DLayer.filterTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  filterTrack mismatch");
                            match = false;
                        }
                        if (!Noedify_Utils.CompareArray<int>(ogLayer.conv3DLayer.channelTrack, newLayer.conv3DLayer.channelTrack))
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  channelTrack mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.no_filters != newLayer.conv3DLayer.no_filters)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  no_filters mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.N_connections_per_node != newLayer.conv3DLayer.N_connections_per_node)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_connections_per_node mismatch");
                            match = false;
                        }
                        if (ogLayer.conv3DLayer.N_weights_per_filter != newLayer.conv3DLayer.N_weights_per_filter)
                        {
                            Debug.Log("CheckNetworkMatch (l=" + l + "):  N_weights_per_filter mismatch");
                            match = false;
                        }
                        break;
                    }
                default: Debug.Log("Warning (Noedify_Manager.CompressedModel.Compress()): Unkown layer case " + ogNet.layers[l].layer_type.ToString()); break;


            }
        }

        return match;
    }

    [System.Serializable]
    public class SavedModel
    {
        public SerializedModel model;

        public string modelName;
        public string dir;
        public string dateCreated;

        public SavedModel(Noedify.Net newModel, string newModelName, System.DateTime creationDate)
        {
            model = new SerializedModel(newModel);
            modelName = newModelName;
            dateCreated = System.DateTime.Now.Year + "-" + System.DateTime.Now.Month + "-" + System.DateTime.Now.Day;
        }

        public void Save(string dir_save = "")
        {
            string fileName = modelName;
            if (string.IsNullOrEmpty(dir_save))
                dir = Application.persistentDataPath;
            else
                dir = dir_save;
            BinaryFormatter bf = new BinaryFormatter();
            FileStream file = File.Create(dir + "/" + fileName + ".dat");

            SavedModel data = this;

            bf.Serialize(file, data);
            file.Close();
            //print("Saved model: " + fileName);
        }

    }

    [System.Serializable]
    public class SavedCompressedModel
    {
        public CompressedModel compressedModel;

        public string modelName;
        public string dir;
        public string dateCreated;

        public SavedCompressedModel(Noedify.Net newModel, string newModelName, System.DateTime creationDate)
        {
            compressedModel = new CompressedModel();
            if (newModel != null)
                compressedModel.Compress(newModel);
            modelName = newModelName;
            dateCreated = System.DateTime.Now.Year + "-" + System.DateTime.Now.Month + "-" + System.DateTime.Now.Day;
        }

        public void Save(string dir_save = "")
        {
            string fileName = modelName;
            if (string.IsNullOrEmpty(dir_save))
                dir = Application.persistentDataPath;
            else
                dir = dir_save;
            BinaryFormatter bf = new BinaryFormatter();
            FileStream file = File.Create(dir + "/" + fileName + ".dat");

            SavedCompressedModel data = this;

            bf.Serialize(file, data);
            file.Close();
            //print("Saved model: " + fileName);

            /*
            // Save as compilable compressed parameter file
            using (var ms = new MemoryStream())
            {
                bf.Serialize(ms, data);
                byte[] modelBytes = ms.ToArray();
                byte[] modelBytes_compressed = CompressData(modelBytes);
                string path = "Assets/Resources/modelBytes.txt";
                StreamWriter writer = new StreamWriter(path, false);
                writer.Write("public static byte[] modelBytes = new byte[" + modelBytes_compressed.Length + "] {");
                for (int i = 0; i < modelBytes_compressed.Length; i++)
                    writer.Write(modelBytes_compressed[i] + ", ");
                writer.WriteLine("};");
                writer.Close();
            }
            */
        }

    }

    static public Noedify.Net Load(string modelName, string dir)
    {
        SavedModel loadModel = new SavedModel(null, "", new System.DateTime());
        if (string.IsNullOrEmpty(dir))
            dir = Application.persistentDataPath;

        string filename;
        if (File.Exists(dir + "/" + modelName + ".dat"))
            filename = dir + "/" + modelName + ".dat";
        else if (File.Exists(dir + "/" + modelName))
            filename = dir + "/" + modelName;
        else
            goto NoFile;

        BinaryFormatter bf = new BinaryFormatter();
        FileStream file = File.Open(filename, FileMode.Open);
        loadModel = (SavedModel)bf.Deserialize(file);
        file.Close();
        return loadModel.model.ReturnNet();

        NoFile:
        return null;
    }

    static public bool LoadCompressedModel(ref Noedify.Net net, string modelName, string dir)
    {

        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();
        SavedCompressedModel loadModel = new SavedCompressedModel(null, "", new System.DateTime());
        if (string.IsNullOrEmpty(dir))
            dir = Application.persistentDataPath;

        string filename;
        if (File.Exists(dir + "/" + modelName + ".dat"))
            filename = dir + "/" + modelName + ".dat";
        else if (File.Exists(dir + "/" + modelName))
            filename = dir + "/" + modelName;
        else
            goto NoFile;

        BinaryFormatter bf = new BinaryFormatter();
        /*
        FileStream file = File.Open(filename, FileMode.Open);
        loadModel = (SavedCompressedModel)bf.Deserialize(file);
        file.Close();
        loadModel.compressedModel.Unwrap(ref net);

        sw.Stop();

        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();
        Debug.Log("load from compressed model time: " + sw.ElapsedMilliseconds + " ms");sw.Reset();sw.Start();
        */

        sw.Start();
        byte[] dataBytes = File.ReadAllBytes(filename);
        sw.Stop();
        Debug.Log("load time uncrompressed time: " + sw.ElapsedMilliseconds + " ms"); sw.Reset();

        using (Stream ms = new MemoryStream(dataBytes))
        {
            loadModel = (SavedCompressedModel)bf.Deserialize(ms);
        }
        sw.Reset(); sw.Start();
        loadModel.compressedModel.Unwrap(ref net);

        sw.Stop();
        Debug.Log("unwrap time: " + sw.ElapsedMilliseconds + " ms");

        return true;
        /*
        byte[] decompressedModel = DecompressData(Noedify_CompileableNetParams.modelBytes);
        using (var ms = new MemoryStream())
        {
            ms.Write(decompressedModel, 0, decompressedModel.Length);
            ms.Seek(0, SeekOrigin.Begin);
            //byte[] model_compressed_bytes = (byte[])System.Convert.ChangeType(bf.Deserialize(ms), typeof(byte[])); // different approach

            loadModel = (SavedCompressedModel)bf.Deserialize(ms);
        }
        loadModel.compressedModel.Unwrap(ref net);

        return true;
        */
        NoFile:
        return false;
    }

    private static byte[] ConvertDataToByeArray(object data)
    {
        BinaryFormatter bf = new BinaryFormatter();
        byte[] outputBytes;
        using (var ms = new MemoryStream())
        {
            bf.Serialize(ms, data);
            outputBytes = ms.ToArray();
        }
        return outputBytes;
    }
    private static void WriteBytesToCompilableFile(byte[] data, string path, bool compress = false)
    {
        using (var ms = new MemoryStream())
        {
            byte[] data_compressed;
            if (compress)
                data_compressed = CompressData(data);
            else
                data_compressed = data;

            StreamWriter writer = new StreamWriter(path, false);
            writer.Write("public static byte[] modelBytes = new byte[" + data_compressed.Length + "] {");
            for (int i = 0; i < data_compressed.Length; i++)
                writer.Write(data_compressed[i] + ", ");
            writer.WriteLine("};");
            writer.Close();

        }
    }

    static public bool ImportNetworkParameters(Noedify.Net net, string filename)
    {
        string path, path_wExt;
        if (Application.platform == RuntimePlatform.Android)
        {
            path = Application.persistentDataPath + "/" + filename + ".noedify";
            path_wExt = Application.persistentDataPath + "/" + filename;
        }
        else
        {
            path = "Assets/Resources/Noedify/Models/" + filename + ".noedify";
            path_wExt = "Assets/Resources/Noedify/Models/" + filename;
        }
        
        if (!System.IO.File.Exists(path))
        {
            if (System.IO.File.Exists(path_wExt))
                path = path_wExt;
            else
            {
                Debug.Log("Parameter read failed. File: " + path + " not found");
                return false;
            }
        }

        StreamReader readFile = new StreamReader(path);
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        for (int l = 1; l < net.LayerCount(); l++)
        //for (int l = 1; l < 2; l++)
        {
            switch (net.layers[l].layer_type)
            {
                case (Noedify.LayerType.Convolutional2D):
                    {

                        int input_channels = 1;
                        if (net.layers[l - 1].layer_type == Noedify.LayerType.Input2D)
                            input_channels = net.layers[l - 1].in_channels;
                        else if (Noedify_Utils.Is2DLayerType(net.layers[l-1]))
                            input_channels = net.layers[l - 1].conv2DLayer.no_filters;
                        else if (Noedify_Utils.Is3DLayerType(net.layers[l-1]))
                            input_channels = net.layers[l-1].conv3DLayer.no_filters;
                        System.Diagnostics.Stopwatch sw1 = new System.Diagnostics.Stopwatch();
                        System.Diagnostics.Stopwatch sw2 = new System.Diagnostics.Stopwatch();
                        

                        sw1.Start();
                        for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                        {
                            sw2.Start();
                            sw.Start();
                            char[] separators = new char[net.layers[l].layerSize];
                            for (int k = 0; k < net.layers[l].layerSize; k++)
                                separators[k] = ',';
                            string weightLine = readFile.ReadLine();
                            //Debug.Log("read line time: " + sw.ElapsedMilliseconds); sw.Reset();
                            sw.Start();
                            string[] weightStrings = weightLine.Split(separators);
                            int expected_weight_count = net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1] * input_channels;
                            if (weightStrings.Length != expected_weight_count)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " kernal size incorrect. Expected " + expected_weight_count + " weights/filter found: " + weightStrings.Length + " weights/filter");
                                return false;
                            }
                            sw.Stop();
                            //Debug.Log("Split time: " + sw.ElapsedMilliseconds);sw.Reset();sw.Start();
                            for (int k = 0; k < net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1] * input_channels; k++)
                            {
                                //Debug.Log("new weight (f=" + f + ", k = " + k + "): " + net.layers[l].weights.valuesConv[f, j]);
                                float.TryParse(weightStrings[k], out net.layers[l].weights.valuesConv[f, k]);
                            }
                            //sw.Stop();Debug.Log("Parse time: " + sw.ElapsedMilliseconds); sw.Reset();

                            sw2.Stop();
                            //Debug.Log("** filter " + f + " weights load time: " + sw2.ElapsedMilliseconds);
                        }
                        sw1.Stop();
                        //Debug.Log("*** Weights load time: " + sw1.ElapsedMilliseconds);
                        if (readFile.ReadLine() != "*")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " layer size incorrect (too many weights)");
                            return false;
                        }
                        for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                        {
                            string biasLine = readFile.ReadLine();
                            if (biasLine == "***")
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " layer size incorrect (too many biases in parameter file)");
                                return false;
                            }
                            else
                            {
                                //Debug.Log("new bias: " + net.layers[l].biases.values[j]);
                                float.TryParse(biasLine, out net.layers[l].biases.valuesConv[f]);
                            }
                        }
                        if (readFile.ReadLine() != "***")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " layer size incorrect (not enough biases in parameter file)");
                            return false;
                        }

                        break;
                    }
                case (Noedify.LayerType.TranspConvolutional2D):
                    {
                        if (Noedify_Utils.Is1DLayerType(net.layers[l - 1])) // Transpose Convolutional 2D (previous layer 1D)
                        {
                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                float[] weightValues = ParseParameterLine(readFile, net.layers[l - 1].layerSize);
                                if (weightValues.Length != net.layers[l - 1].layerSize)
                                {
                                    Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") previous layer size incorrect. Expected: " + net.layers[l - 1].layerSize + " weights, found: " + weightValues.Length + " weights.");
                                    return false;
                                }
                                for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                                    net.layers[l].weights.values[i, j] = weightValues[i];
                            }
                            if (readFile.ReadLine() != "*")
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (not enough weights in parameter file)");
                                return false;
                            }

                            float[] biasValues = ParseParameterLine(readFile, net.layers[l].conv2DLayer.no_filters);
                            bool skipBias = false;
                            if (biasValues == null || biasValues.Length == 0)
                            {
                                Debug.Log("No bias parameters detected for layer " + l + ", skipping to next layer");
                                skipBias = true;
                            }
                            else
                            {
                                if (biasValues.Length != net.layers[l].conv2DLayer.no_filters)
                                {
                                    Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") previous layer size incorrect. Expected: " + net.layers[l].conv2DLayer.no_filters + " biases, found: " + biasValues.Length + " biases.");
                                    return false;
                                }
                                for (int i = 0; i < net.layers[l].in_channels; i++)
                                {
                                    net.layers[l].biases.valuesConv[i] = biasValues[i];
                                }
                            }
                            if (!skipBias)
                            {
                                if (readFile.ReadLine() != "***")
                                {
                                    Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (not enough biases in parameter file)");
                                    return false;
                                }
                            }
                        }
                        else // Transpose Convolutional 2D (previous layer 2D)
                        {
                            for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                            {
                                float[] weightValues = ParseParameterLine(readFile, net.layers[l].conv2DLayer.N_weights_per_filter);
                                if (weightValues.Length != net.layers[l].conv2DLayer.N_weights_per_filter)
                                {
                                    Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") previous layer size incorrect. Expected: " + net.layers[l].conv2DLayer.N_weights_per_filter + " weights, found: " + weightValues.Length + " weights.");
                                    return false;
                                }
                                for (int j = 0; j < net.layers[l].conv2DLayer.N_weights_per_filter; j++)
                                {
                                    net.layers[l].weights.valuesConv[f, j] = weightValues[j];

                                }
                            }

                            string endLine = readFile.ReadLine();
                            if (endLine != "*")
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (too many weights). Expected " + net.layers[l].conv2DLayer.no_filters + " lines");
                                Debug.Log("Extra parameter line: " + endLine);
                                return false;
                            }

                            float[] biasValues = ParseParameterLine(readFile, net.layers[l].in_channels);
                            bool skipBias = false;
                            if (biasValues == null || biasValues.Length == 0)
                            {
                                Debug.Log("No bias parameters detected for layer " + l + ", skipping to next layer");
                                skipBias = true;
                            }
                            else
                            {
                                if (biasValues.Length != net.layers[l].conv2DLayer.no_filters)
                                {
                                    Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") previous layer size incorrect. Expected: 1 biases/filter, found: " + biasValues.Length + " biases/filter.");
                                    return false;
                                }
                                for (int i = 0; i < net.layers[l].in_channels; i++)
                                    net.layers[l].biases.valuesConv[i] = biasValues[i];
                            }
                            if (!skipBias)
                            {
                                if (readFile.ReadLine() != "***")
                                {
                                    Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (not enough biases in parameter file)");
                                    return false;
                                }
                            }
                        }
                        break;
                    }
                case (Noedify.LayerType.BatchNorm2D):
                    {
                        // load weights
                        float[] weightValues = ParseParameterLine(readFile, net.layers[l].in_channels);
                        if (weightValues.Length != net.layers[l].in_channels)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") kernel count incorrect. Expected: " + net.layers[l].in_channels + " weights, found: " + weightValues.Length + " weights.");
                            return false;
                        }
                        for (int i = 0; i < net.layers[l].in_channels; i++)
                        {
                            float weightVal = weightValues[i];
                            float assignVal = net.layers[l].weights.values[0, i];
                            net.layers[l].weights.values[0, i] = weightValues[i];
                        }
                        // load biases
                        if (readFile.ReadLine() != "*")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (missing weights)");
                            return false;
                        }

                        float[] biasValues = ParseParameterLine(readFile, net.layers[l].in_channels);
                        bool skipBias = false;
                        if (biasValues == null || biasValues.Length == 0)
                        {
                            Debug.Log("No bias parameters detected for layer " + l + ", skipping to next layer");
                            skipBias = true;
                        }
                        else
                        {
                            if (biasValues.Length != net.layers[l].in_channels)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") kernal count incorrect. Expected: " + net.layers[l].in_channels + " biases, found: " + biasValues.Length + " biases.");
                                return false;
                            }
                            for (int i = 0; i < net.layers[l].in_channels; i++)
                            {
                                net.layers[l].biases.values[i] = biasValues[i];
                            }
                        }
                        // Load running mean
                        if (readFile.ReadLine() != "*")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (missing running mean)");
                            return false;
                        }
                        
                        float[] runningMeanValues = ParseParameterLine(readFile, net.layers[l].in_channels);
                        if (runningMeanValues == null || runningMeanValues.Length == 0)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") no parameters found for \"running mean\"");
                            return false;
                        }
                        else
                        {
                            if (runningMeanValues.Length != net.layers[l].in_channels)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") kernel count incorrect. Expected: " + net.layers[l].in_channels + " running mean, found: " + runningMeanValues.Length + " running mean.");
                                return false;
                            }
                            for (int i = 0; i < net.layers[l].in_channels; i++)
                            {
                                net.layers[l].bn_running_mean[i] = runningMeanValues[i];
                            }
                        }
                        // Load running variance
                        if (readFile.ReadLine() != "*")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (missing running variance)");
                            return false;
                        }
                        
                        float[] runningVarValues = ParseParameterLine(readFile, net.layers[l].in_channels);
                        if (runningVarValues == null || runningVarValues.Length == 0)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") no parameters found for \"running variance\"");
                            return false;
                        }
                        else
                        {
                            if (runningVarValues.Length != net.layers[l].in_channels)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") kernel count incorrect. Expected: " + net.layers[l].in_channels + " running var, found: " + runningVarValues.Length + " running var.");
                                return false;
                            }
                            for (int i = 0; i < net.layers[l].in_channels; i++)
                            {
                                net.layers[l].bn_running_var[i] = runningVarValues[i];
                            }
                        }


                        if (!skipBias)
                        {
                            if (readFile.ReadLine() != "***")
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") EOF not detected (***)");
                                return false;
                            }
                        }
                        break;
                    }
                case (Noedify.LayerType.FullyConnected):
                case (Noedify.LayerType.Output):
                    {
                        for (int j = 0; j < net.layers[l].layerSize; j++)
                        {
                            char[] separators = new char[net.layers[l].layerSize];
                            for (int k = 0; k < net.layers[l].layerSize; k++)
                                separators[k] = ',';
                            string weightLine = readFile.ReadLine();
                            string[] weightStrings = weightLine.Split(separators);
                            if (weightStrings.Length != net.layers[l - 1].layerSize)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") previous layer size incorrect. Expected: " + net.layers[l - 1].layerSize + " weights, found: " + weightStrings.Length + " weights.");
                                return false;
                            }
                            for (int k = 0; k < net.layers[l - 1].layerSize; k++)
                            {
                                float.TryParse(weightStrings[k], out net.layers[l].weights.values[k, j]);
                                //Debug.Log("new weight: " + net.layers[l].weights.values[k, j]);
                            }
                        }
                        if (readFile.ReadLine() != "*")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (not enough weights in parameter file)");
                            return false;
                        }
                        for (int j = 0; j < net.layers[l].layerSize; j++)
                        {
                            string biasLine = readFile.ReadLine();
                            if (biasLine == "***")
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (too many biases in parameter file)");
                                return false;
                            }
                            else
                            {
                                float.TryParse(biasLine, out net.layers[l].biases.values[j]);
                                //Debug.Log("new bias: " + net.layers[l].biases.values[j]);
                            }
                        }
                        if (readFile.ReadLine() != "***")
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer size incorrect (not enough biases in parameter file)");
                            return false;
                        }
                        break;
                    }
                default:
                    {
                        Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer type (" + net.layers[l].layer_type.ToString() + ") incompatable");
                        return false;
                    }
            }

            //net.layers[l].weights = newLayerWeights;
            //net.layers[l].biases = newLayerBiases;

        }
        return true;
    }

    public static bool LoadModelConfiguration(ref Noedify.Net model, string configCode)
    {
        string[] temp = configCode.Split('_');
        if (temp.Length < 5)
        {
            Debug.Log("Invalid config code (need at least 5 sections, found: " + temp.Length + "): " + configCode);
            goto Error;
        }
        int no_layers = temp[2].Length;
        Debug.Log("reading " + no_layers + " layers");
        List<Noedify.Layer> layerList = new List<Noedify.Layer>();
        List<Noedify.ActivationFunction> activationFunctionList = new List<Noedify.ActivationFunction>();
        List<string> parameterStrings = new List<string>();


        // Input Layer
        Noedify.Layer inputLayer = new Noedify.Layer(Noedify.LayerType.Input, 1);
        switch (temp[0]){
            case ("1"):
                {
                    // Input 1D
                    int layerSize = 0;
                    if (int.TryParse(temp[1], out layerSize))
                        inputLayer = new Noedify.Layer(Noedify.LayerType.Input, layerSize, "input1D");
                    else
                    {
                        Debug.Log("Unable to parse parameters for layer input1D: " + temp[1]);
                        goto Error;
                    }
                    break;
                }
            case ("2"):
                {
                    // Input 2D
                    string[] paramTemp = temp[1].Split(',');
                    if (paramTemp.Length == 3)
                    {
                        int layerSize_x = 0;
                        int layerSize_y = 0;
                        int no_ch = 0;
                        if (!int.TryParse(paramTemp[0], out layerSize_x))
                        {
                            Debug.Log("Unable to parse parameters for layer input2D: " + paramTemp[0]);
                            goto Error;
                        }
                        if (!int.TryParse(paramTemp[1], out layerSize_y))
                        {
                            Debug.Log("Unable to parse parameters for layer input2D: " + paramTemp[1]);
                            goto Error;
                        }
                        if (!int.TryParse(paramTemp[2], out no_ch))
                        {
                            Debug.Log("Unable to parse parameters for layer input2D: " + paramTemp[2]);
                            goto Error;
                        }

                        inputLayer = new Noedify.Layer(
                            newType: Noedify.LayerType.Input2D,
                            inSize: new int[2] { layerSize_x, layerSize_y },
                            noChannels: no_ch,
                            newLayerName: "input2D"
                            );

                        break;
                    }
                    else
                    {
                        Debug.Log("Unable to parse parameters for layer input2D (expected 3 parameters, found: " + paramTemp.Length + "): " + temp[1]);
                        goto Error;
                    }

                }
            case ("3"):
                {
                    // Input 3D
                    string[] paramTemp = temp[1].Split(',');
                    if (paramTemp.Length == 4)
                    {
                        int layerSize_x = 0;
                        int layerSize_y = 0;
                        int layerSize_z = 0;
                        int no_ch = 0;
                        if (!int.TryParse(paramTemp[0], out layerSize_x))
                        {
                            Debug.Log("Unable to parse parameters for layer input3D: " + paramTemp[0]);
                            goto Error;
                        }
                        if (!int.TryParse(paramTemp[1], out layerSize_y))
                        {
                            Debug.Log("Unable to parse parameters for layer input3D: " + paramTemp[1]);
                            goto Error;
                        }
                        if (!int.TryParse(paramTemp[2], out layerSize_z))
                        {
                            Debug.Log("Unable to parse parameters for layer input3D: " + paramTemp[2]);
                            goto Error;
                        }
                        if (!int.TryParse(paramTemp[3], out no_ch))
                        {
                            Debug.Log("Unable to parse parameters for layer input3D: " + paramTemp[3]);
                            goto Error;
                        }

                        inputLayer = new Noedify.Layer(
                            newType: Noedify.LayerType.Input3D,
                            inSize: new int[3] { layerSize_x, layerSize_y, layerSize_z},
                            noChannels: no_ch,
                            newLayerName: "input3D"
                            );

                        break;
                    }
                    else
                    {
                        Debug.Log("Unable to parse parameters for layer input3D (expected 4 parameters, found: " + paramTemp.Length + "): " + temp[1]);
                        goto Error;
                    }

                }
            default: { Debug.Log("ERROR (LoadModelConfiguration): invalid input layer type code: " + temp[0]); goto Error; }

        }
        // Load layer parameters strings
        for (int n = 0; n < no_layers; n++)
        {
            parameterStrings.Add(temp[3 + n]);
        }
        // load activation function code
        for (int n = 0; n < no_layers; n++)
        {
            switch (temp[temp.Length-1][n])
            {
                case ('L'):
                    {
                        // linear
                        activationFunctionList.Add(Noedify.ActivationFunction.Linear);
                        break;
                    }
                case ('R'):
                    {
                        // relu
                        activationFunctionList.Add(Noedify.ActivationFunction.ReLU);
                        break;
                    }
                case ('S'):
                    {
                        // sigmoid
                        activationFunctionList.Add(Noedify.ActivationFunction.Sigmoid);
                        break;
                    }
                case ('T'):
                    {
                        // tanh
                        activationFunctionList.Add(Noedify.ActivationFunction.Tanh);
                        break;
                    }
                default: { Debug.Log("ERROR (LoadModelConfiguration): invalid activation function code: " + temp[temp.Length - 1][n]); goto Error; }
            }
        }
        // load layer type code and add to list
        Noedify.Layer previousLayer = inputLayer;
        for (int n = 0; n < no_layers; n++)
        {
            Noedify.Layer newLayer = new Noedify.Layer(Noedify.LayerType.FullyConnected, 1);
            switch (temp[2][n])
            {
                case ('a'):
                    {
                        // fully-connected
                        int layerSize = -1;
                        if (int.TryParse(parameterStrings[n], out layerSize))
                            newLayer = new Noedify.Layer(Noedify.LayerType.FullyConnected, layerSize, activationFunctionList[n], "Fully Connected");
                        else
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }
                        break;
                    }
                case ('b'):
                    {
                        // convolutional 2D
                        bool paramLoadStatus = true;
                        string[] layerParams = parameterStrings[n].Split(',');

                        int filterSize_x = 0, filterSize_y = 0, n_ch = 0, stride_x = 0, stride_y = 0, pad_x = 0, pad_y=0 ;

                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[0], out filterSize_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[1], out filterSize_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[2], out n_ch);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[3], out stride_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[4], out stride_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[5], out pad_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[6], out pad_y);

                        if (!paramLoadStatus)
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }

                        newLayer = new Noedify.Layer(
                        newType: Noedify.LayerType.Convolutional2D,
                        previousLayer: previousLayer,
                        filtsize: new int[2] { filterSize_x, filterSize_y },
                        strd: new int[2] { stride_x, stride_y },
                        nfilters: n_ch,
                        pdding: new int[2] { pad_x, pad_y },
                        actFunction: activationFunctionList[n],
                        newLayerName: ("conv2D")
                        );
                        break;
                    }
                case ('c'):
                    {
                        // batchnorm 2D
                        float eps = 0;
                        if (float.TryParse(parameterStrings[n], out eps))
                        {
                            newLayer = new Noedify.Layer(
                                Noedify.LayerType.BatchNorm2D,
                                previousLayer: previousLayer,
                                epsilon: eps,
                                actFunction: activationFunctionList[n],
                                newLayerName: "BatchNorm2D"
                                );
                        }
                        else
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }
                        break;
                    }
                case ('d'):
                    {
                        // convolutional 3D
                        bool paramLoadStatus = true;
                        string[] layerParams = parameterStrings[n].Split(',');

                        int filterSize_x = 0, filterSize_y = 0, n_ch = 0, stride_x = 0, stride_y = 0, pad_x = 0, pad_y = 0;

                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[0], out filterSize_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[1], out filterSize_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[2], out n_ch);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[3], out stride_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[4], out stride_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[5], out pad_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[6], out pad_y);
                        if (!paramLoadStatus)
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }
                        newLayer = new Noedify.Layer(
                            Noedify.LayerType.TranspConvolutional2D,
                            previousLayer: previousLayer,
                            filtsize: new int[2] { filterSize_x, filterSize_y },
                            strd: new int[2] { stride_x, stride_y },
                            nfilters: n_ch,
                            pdding: new int[2] { pad_x, pad_y },
                            actFunction: activationFunctionList[n],
                            newLayerName: "TConv2D");
                        break;
                    }
                case ('e'):
                    {
                        // convolutional 3D
                        bool paramLoadStatus = true;
                        string[] layerParams = parameterStrings[n].Split(',');

                        int filterSize_x = 0, filterSize_y = 0, filterSize_z = 0, n_ch = 0,
                            stride_x = 0, stride_y = 0, stride_z = 0, pad_x = 0, pad_y = 0, pad_z = 0;

                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[0], out filterSize_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[1], out filterSize_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[2], out filterSize_z);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[3], out n_ch);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[4], out stride_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[5], out stride_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[6], out stride_z);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[7], out pad_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[8], out pad_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[9], out pad_z);
                        if (!paramLoadStatus)
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }
                        newLayer = new Noedify.Layer(
                            Noedify.LayerType.Convolutional3D,
                            previousLayer: previousLayer,
                            filtsize: new int[3] { filterSize_x, filterSize_y, filterSize_z },
                            strd: new int[3] { stride_x, stride_y, stride_z },
                            nfilters: n_ch,
                            pdding: new int[3] { pad_x, pad_y, pad_z },
                            actFunction: activationFunctionList[n],
                            newLayerName: "Conv3D");
                        break;
                    }
                case ('f'):
                    {
                        // batchnorm 3D
                        float eps = 0;
                        if (float.TryParse(parameterStrings[n], out eps))
                        {
                            newLayer = new Noedify.Layer(
                                Noedify.LayerType.BatchNorm3D,
                                previousLayer: previousLayer,
                                epsilon: eps,
                                actFunction: activationFunctionList[n],
                                newLayerName: "BatchNorm3D"
                                );
                        }
                        else
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }
                        break;
                    }
                case ('g'):
                    {
                        // transpose convolutional 3D
                        bool paramLoadStatus = true;
                        string[] layerParams = parameterStrings[n].Split(',');

                        int filterSize_x = 0, filterSize_y = 0, filterSize_z = 0, n_ch = 0,
                            stride_x = 0, stride_y = 0, stride_z = 0, pad_x = 0, pad_y = 0, pad_z = 0;

                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[0], out filterSize_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[1], out filterSize_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[2], out filterSize_z);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[3], out n_ch);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[4], out stride_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[5], out stride_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[6], out stride_z);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[7], out pad_x);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[8], out pad_y);
                        paramLoadStatus = paramLoadStatus & int.TryParse(layerParams[9], out pad_z);
                        if (!paramLoadStatus)
                        {
                            Debug.Log("Unable to parse parameters for layer " + n + ": " + parameterStrings[n]);
                            goto Error;
                        }
                        newLayer = new Noedify.Layer(
                            Noedify.LayerType.TranspConvolutional3D,
                            previousLayer: previousLayer,
                            filtsize: new int[3] { filterSize_x, filterSize_y, filterSize_z },
                            strd: new int[3] { stride_x, stride_y, stride_z },
                            nfilters: n_ch,
                            pdding: new int[3] { pad_x, pad_y, pad_z },
                            actFunction: activationFunctionList[n],
                            newLayerName: "TConv3D");
                        break;
                    }
                default: { Debug.Log("ERROR (LoadModelConfiguration): invalid layer type code: " + temp[0][n]); goto Error; }
            }
            layerList.Add(newLayer);
            previousLayer = newLayer;
        }
        model.AddLayer(inputLayer);
        for (int n = 0; n < layerList.Count; n++)
            model.AddLayer(layerList[n]);
        return true;
        Error:
        return false;
    }

#if USECOMPILEABLEPARAMS
    static public bool ImportNetworkCompilableParameters(Noedify.Net net)
    {
        Noedify_CompileableNetParams.NetParam[] netParams = Noedify_CompileableNetParams.netParams;

        if (netParams.Length != net.LayerCount())
        {
            Debug.LogError("Parameter failed. Expected parameter file with " + net.LayerCount() + " layers but found " + netParams.Length + " layers.");
            return false;
        }

        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

        for (int l = 1; l < net.LayerCount(); l++)
        {
            //Debug.Log("Importing layer: " + l);
            float[,] weights = netParams[l].weight;
            float[] biases = netParams[l].bias;
            switch (net.layers[l].layer_type)
            {
                case (Noedify.LayerType.Convolutional2D):
                    {

                        int input_channels = 1;
                        if (net.layers[l - 1].layer_type == Noedify.LayerType.Input2D)
                            input_channels = net.layers[l - 1].in_channels;
                        else if (net.layers[l - 1].layer_type == Noedify.LayerType.Convolutional2D | net.layers[l - 1].layer_type == Noedify.LayerType.Pool2D | net.layers[l - 1].layer_type == Noedify.LayerType.BatchNorm2D)
                            input_channels = net.layers[l - 1].conv2DLayer.no_filters;
                        System.Diagnostics.Stopwatch sw1 = new System.Diagnostics.Stopwatch();
                        System.Diagnostics.Stopwatch sw2 = new System.Diagnostics.Stopwatch();

                        int size1 = weights.GetLength(0);
                        int size2 = weights.GetLength(1);
                        if (size2 != net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1] * input_channels | size1 != net.layers[l].conv2DLayer.no_filters)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].conv2DLayer.no_filters + ", " + (net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1] * input_channels) + ") found weight shape: (" + size1 + ", " + size2 + ")");
                            return false;
                        }
                        for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                            for (int k = 0; k < net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1] * input_channels; k++)
                                net.layers[l].weights.valuesConv[f, k] = weights[f, k];

                        size1 = biases.Length;
                        if (size1 != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + size1 + ")");
                            return false;
                        }

                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.valuesConv[j] = biases[j];

                        break;
                    }
                case (Noedify.LayerType.Convolutional3D):
                    {

                        int input_channels = 1;
                        if (net.layers[l - 1].layer_type == Noedify.LayerType.Input3D)
                            input_channels = net.layers[l - 1].in_channels;
                        else if (net.layers[l - 1].layer_type == Noedify.LayerType.Convolutional3D | net.layers[l - 1].layer_type == Noedify.LayerType.TranspConvolutional3D | net.layers[l - 1].layer_type == Noedify.LayerType.BatchNorm3D)
                            input_channels = net.layers[l - 1].conv3DLayer.no_filters;
                        System.Diagnostics.Stopwatch sw1 = new System.Diagnostics.Stopwatch();
                        System.Diagnostics.Stopwatch sw2 = new System.Diagnostics.Stopwatch();

                        int size1 = weights.GetLength(0);
                        int size2 = weights.GetLength(1);
                        if (size2 != net.layers[l].conv3DLayer.filterSize[0] * net.layers[l].conv3DLayer.filterSize[1] * net.layers[l].conv3DLayer.filterSize[2] * input_channels | size1 != net.layers[l].conv3DLayer.no_filters)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].conv3DLayer.no_filters + ", " + (net.layers[l].conv3DLayer.filterSize[0] * net.layers[l].conv3DLayer.filterSize[1] * net.layers[l].conv3DLayer.filterSize[2] * input_channels) + ") found weight shape: (" + size1 + ", " + size2 + ")");
                            return false;
                        }
                        for (int f = 0; f < net.layers[l].conv3DLayer.no_filters; f++)
                            for (int k = 0; k < net.layers[l].conv3DLayer.filterSize[0] * net.layers[l].conv3DLayer.filterSize[1] * net.layers[l].conv3DLayer.filterSize[2] * input_channels; k++)
                                net.layers[l].weights.valuesConv[f, k] = weights[f, k];

                        size1 = biases.Length;
                        if (size1 != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + size1 + ")");
                            return false;
                        }

                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.valuesConv[j] = biases[j];

                        break;
                    }
                case (Noedify.LayerType.TranspConvolutional2D):
                    {
                        if (Noedify_Utils.Is1DLayerType(net.layers[l - 1])) // Transpose Convolutional 2D (previous layer 1D)
                        {
                            int size1 = weights.GetLength(0);
                            int size2 = weights.GetLength(1);
                            if (size2 != net.layers[l].layerSize | size1 != net.layers[l - 1].layerSize)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].layerSize + ", " + net.layers[l - 1].layerSize + ") found weight shape: (" + size1 + ", " + size2 + ")");
                                return false;
                            }

                            for (int j = 0; j < net.layers[l].layerSize; j++)
                                for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                                    net.layers[l].weights.values[i, j] = weights[i, j];
                        }
                        else // Transpose Convolutional 2D (previous layer 2D)
                        {
                            int size1 = weights.GetLength(0);
                            int size2 = weights.GetLength(1);
                            if (size1 != net.layers[l].conv2DLayer.no_filters | size2 != net.layers[l].conv2DLayer.N_weights_per_filter)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].conv2DLayer.no_filters + ", " + net.layers[l].conv2DLayer.N_weights_per_filter + ") found weight shape: (" + size1 + ", " + size2 + ")");
                                return false;
                            }

                            for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                                for (int j = 0; j < net.layers[l].conv2DLayer.N_weights_per_filter; j++)
                                {
                                    net.layers[l].weights.valuesConv[f, j] = weights[f, j];
                                }
                        }
                        int sizeB = biases.Length;
                        if (sizeB != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + sizeB + ")");
                            return false;
                        }
                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.valuesConv[j] = biases[j];
                        break;
                    }
                case (Noedify.LayerType.TranspConvolutional3D):
                    {
                        if (Noedify_Utils.Is1DLayerType(net.layers[l - 1])) // Transpose Convolutional 3D (previous layer 1D)
                        {
                            int size1 = weights.GetLength(0);
                            int size2 = weights.GetLength(1);
                            if (size2 != net.layers[l].layerSize | size1 != net.layers[l - 1].layerSize)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].layerSize + ", " + net.layers[l - 1].layerSize + ") found weight shape: (" + size1 + ", " + size2 + ")");
                                return false;
                            }

                            for (int j = 0; j < net.layers[l].layerSize; j++)
                                for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                                    net.layers[l].weights.values[i, j] = weights[i, j];
                        }
                        else // Transpose Convolutional 3D (previous layer 3D)
                        {
                            int size1 = weights.GetLength(0);
                            int size2 = weights.GetLength(1);
                            if (size1 != net.layers[l].conv3DLayer.no_filters | size2 != net.layers[l].conv3DLayer.N_weights_per_filter)
                            {
                                Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].conv2DLayer.no_filters + ", " + net.layers[l].conv2DLayer.N_weights_per_filter + ") found weight shape: (" + size1 + ", " + size2 + ")");
                                return false;
                            }

                            for (int f = 0; f < net.layers[l].conv3DLayer.no_filters; f++)
                                for (int j = 0; j < net.layers[l].conv3DLayer.N_weights_per_filter; j++)
                                {
                                    net.layers[l].weights.valuesConv[f, j] = weights[f, j];
                                }
                        }
                        int sizeB = biases.Length;
                        if (sizeB != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + sizeB + ")");
                            return false;
                        }
                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.valuesConv[j] = biases[j];
                        break;
                    }
                case (Noedify.LayerType.BatchNorm2D):
                    {
                        int size1 = weights.GetLength(0);
                        int size2 = weights.GetLength(1);
                        if (size1 != 1 | size2 != net.layers[l].in_channels)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + 1 + ", " + net.layers[l].in_channels + ") found weight shape: (" + size1 + ", " + size2 + ")");
                            return false;
                        }

                        for (int i = 0; i < net.layers[l].in_channels; i++)
                            net.layers[l].weights.values[0, i] = weights[0, i];

                        int sizeB = biases.Length;
                        if (sizeB != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + sizeB + ")");
                            return false;
                        }
                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.values[j] = biases[j];


                        break;
                    }
                case (Noedify.LayerType.BatchNorm3D):
                    {
                        int size1 = weights.GetLength(0);
                        int size2 = weights.GetLength(1);
                        if (size1 != 1 | size2 != net.layers[l].in_channels)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + 1 + ", " + net.layers[l].in_channels + ") found weight shape: (" + size1 + ", " + size2 + ")");
                            return false;
                        }

                        for (int i = 0; i < net.layers[l].in_channels; i++)
                            net.layers[l].weights.values[0, i] = weights[0, i];

                        int sizeB = biases.Length;
                        if (sizeB != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + sizeB + ")");
                            return false;
                        }
                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.values[j] = biases[j];
                        break;
                    }
                case (Noedify.LayerType.FullyConnected):
                case (Noedify.LayerType.Output):
                    {
                        int size1 = weights.GetLength(0);
                        int size2 = weights.GetLength(1);
                        if (size2 != net.layers[l].layerSize | size1 != net.layers[l - 1].layerSize)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") weight size incorrect. Expected shape (" + net.layers[l].layerSize + ", " + net.layers[l - 1].layerSize + ") found weight shape: (" + size1 + ", " + size2 + ")");
                            return false;
                        }

                        for (int j = 0; j < net.layers[l].layerSize; j++)
                            for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                                net.layers[l].weights.values[i, j] = weights[i, j];

                        int sizeB = biases.Length;
                        if (sizeB != net.layers[l].no_biases)
                        {
                            Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") bias size incorrect. Expected shape (" + net.layers[l].no_biases + ") found bias shape: (" + sizeB + ")");
                            return false;
                        }
                        for (int j = 0; j < net.layers[l].no_biases; j++)
                            net.layers[l].biases.valuesConv[j] = biases[j];

                        break;
                    }
                default:
                    {
                        Debug.Log("Import network parameters failed: layer " + l + " (" + net.layers[l].name + ") layer type (" + net.layers[l].layer_type.ToString() + ") incompatable");
                        return false;
                    }
            }
        }
        return true;
    }
#endif

#if NOEDIFY_NORELEASE
    static public List<Noedify.Layer> SparsifyNetwork(Noedify.Net net, float weight_threshold)
    {
        List<Noedify.Layer> newLayerList = new List<Noedify.Layer>();
        for (int l = 0; l < net.LayerCount(); l++)
        {
            int N_sparse_parameters = 0;

            Noedify.Layer sparseLayer = new Noedify.Layer(Noedify.LayerType.FullyConnected, 1, "");

            switch (net.layers[l].layer_type)
            {
                case (Noedify.LayerType.Input):
                case (Noedify.LayerType.Input2D):
                    {
                        sparseLayer = net.layers[l];
                        break;
                    }
                case (Noedify.LayerType.Output):
                case (Noedify.LayerType.FullyConnected):
                    {
                        sparseLayer = net.layers[l];
                        for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                        {
                            for (int j = 0; j < net.layers[l].layerSize; j++)
                            {
                                if (Mathf.Abs(sparseLayer.weights.values[i, j]) < weight_threshold)
                                {
                                    sparseLayer.weights.values[i, j] = 0;
                                    N_sparse_parameters++;
                                }
                            }
                        }
                        break;
                    }
                case (Noedify.LayerType.BatchNorm2D):
                    {
                        sparseLayer = net.layers[l];
                        break;
                    }
                case (Noedify.LayerType.TranspConvolutional2D):
                    {
                        sparseLayer = net.layers[l];
                        if (Noedify_Utils.Is1DLayerType(net.layers[l - 1]))
                        {
                            for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                            {
                                for (int j = 0; j < net.layers[l].layerSize; j++)
                                {
                                    if (Mathf.Abs(sparseLayer.weights.values[i, j]) < weight_threshold)
                                    {
                                        sparseLayer.weights.values[i, j] = 0;
                                        N_sparse_parameters++;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                            {
                                int nd = net.layers[net.layers[l].layer_no - 1].conv2DLayer.nodeTrack[i];
                                for (int conn = 0; conn < net.layers[l].conv2DLayer.N_connections_per_node; conn++)
                                {
                                    int connected_j = net.layers[l].conv2DLayer.connections[nd, conn];
                                    int cif = net.layers[l].conv2DLayer.connectionsInFilter[nd, conn] + net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1] * net.layers[net.layers[l].layer_no - 1].conv2DLayer.filterTrack[i];

                                    if (connected_j > -1)
                                    {
                                        if (Mathf.Abs(net.layers[l].weights.valuesConv[net.layers[l].conv2DLayer.filterTrack[connected_j], cif]) < weight_threshold)
                                        {
                                            //net.layers[l].conv2DLayer.connections[i, conn] = -1;
                                            net.layers[l].weights.valuesConv[net.layers[l].conv2DLayer.filterTrack[connected_j], cif] = 0;
                                            N_sparse_parameters++;
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                default:
                    {
                        Debug.Log("Warning (Noedify_Manager.SparsifyNetwork()): Unkown layer case " + net.layers[l].layer_type.ToString());
                        sparseLayer = net.layers[l];
                        break;
                    }
            }
            newLayerList.Add(sparseLayer);
            Debug.Log("Layer " + l + ": Sparsified " + N_sparse_parameters + " parameters");
        }
        return newLayerList;
    }

    public static void WriteOutputToFile(string filename, int no_ch, int ch_sz_x, int ch_sz_y, int ch_sz_z, float[] values, string variableName = "output")
    {
        string path = "Assets/Resources/" + filename;

        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine(variableName + " = {");
        for (int c = 0; c < no_ch; c++)
        {
            string outString = "{";
            for (int iz = 0; iz < ch_sz_z; iz++)
            {
                outString += "[";
                for (int iy = 0; iy < ch_sz_y; iy++)
                {
                    for (int ix = 0; ix < ch_sz_x; ix++)
                    {
                        outString += values[c * ch_sz_z * ch_sz_y * ch_sz_x + iz * ch_sz_y * ch_sz_x + iy * ch_sz_x + ix] + ", ";
                    }
                    outString += "; ";
                }
                outString += "], ";
            }
            outString += "},";
            writer.WriteLine(outString);
        }
        writer.WriteLine("};");
        //writer.WriteLine("clc;");
        writer.Close();
        Debug.Log("Created file: " + path);
    }

    public static void WriteOutputToFile(string filename, int no_ch, int ch_sz_x, int ch_sz_y, int ch_sz_z, float[,,,] values, string variableName = "output")
    {
        string path = "Assets/Resources/" + filename;

        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine(variableName + " = {");
        for (int c = 0; c < no_ch; c++)
        {
            string outString = "{";
            for (int iz = 0; iz < ch_sz_z; iz++)
            {
                outString += "[";
                for (int iy = 0; iy < ch_sz_y; iy++)
                {
                    for (int ix = 0; ix < ch_sz_x; ix++)
                    {
                        outString += values[c, ix, iy, iz] + ", ";
                    }
                    outString += "; ";
                }
                outString += "], ";
            }
            outString += "},";
            writer.WriteLine(outString);
        }
        writer.WriteLine("};");
        //writer.WriteLine("clc;\n");
        writer.Close();
        Debug.Log("Created file: " + path);
    }

    public static void WriteOutputToFile(string filename, int no_ch, int ch_sz_x, int ch_sz_y, float[] values, string variableName = "output")
    {
        string path = "Assets/Resources/" + filename;

        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine(variableName + " = {");
        for (int c = 0; c < no_ch; c++)
        {
            string outString = "[";
            for (int iy = 0; iy < ch_sz_y; iy++)
            {
                for (int ix = 0; ix < ch_sz_x; ix++)
                {
                    outString += values[c * ch_sz_y * ch_sz_x + iy * ch_sz_x + ix] + ", ";
                }
                outString += "; ";
            }
            outString += "], ";
            writer.WriteLine(outString);
        }
        writer.WriteLine("};");
        writer.Close();
        Debug.Log("Created file: " + path);
    }

    public static void WriteOutputToFile(string filename, int no_ch, int ch_sz, float[,,] values)
    {
        string path = "Assets/Resources/" + filename;

        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine("unpred = {");
        for (int c = 0; c < no_ch; c++)
        {
            string outString = "[";
            for (int iy = 0; iy < ch_sz; iy++)
            {
                for (int ix = 0; ix < ch_sz; ix++)
                {
                    outString += values[c, iy, ix] + ", ";
                }
                outString += "; ";
            }
            outString += "], ";
            writer.WriteLine(outString);
        }
        writer.WriteLine("};");
        writer.Close();
    }

#endif
    static public void ExportNetworkParameters(Noedify.Net net, string fileName)
    {
        string path = "Assets/Resources/NodeAI/ExportedModels/" + fileName + ".txt";

        StreamWriter writeFile = new StreamWriter(path, false);

        for (int l = 1; l < net.LayerCount(); l++)
        {
            //writeFile.WriteLine("layer " + l + ": " + net.layers[l].name + " type " + net.layers[l].layer_type.ToString());
            if (net.layers[l].layer_type == Noedify.LayerType.Convolutional2D)
            {
                if (net.layers[l - 1].layer_type == Noedify.LayerType.Input2D)
                {
                    for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                    {
                        string weightLine = "";
                        for (int j = 0; j < net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1]; j++)
                            weightLine += net.layers[l].weights.valuesConv[f, j] + ",";
                        weightLine = weightLine.Remove(weightLine.Length - 1);
                        writeFile.WriteLine(weightLine);
                    }
                    writeFile.WriteLine("*");
                    //string biasLine = "";
                    for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                    {
                        writeFile.WriteLine(net.layers[l].biases.valuesConv[f]);
                    }
                    //biasLine = biasLine.Remove(biasLine.Length - 2);
                    // writeFile.WriteLine(biasLine);
                }
                else if (net.layers[l - 1].layer_type == Noedify.LayerType.Convolutional2D)
                {
                    int channels = net.layers[l - 1].conv2DLayer.no_filters;
                    for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                    {
                        string weightLine = "";
                        for (int j = 0; j < channels * net.layers[l].conv2DLayer.filterSize[0] * net.layers[l].conv2DLayer.filterSize[1]; j++)
                            weightLine += net.layers[l].weights.valuesConv[f, j] + ",";
                        weightLine = weightLine.Remove(weightLine.Length - 1);
                        writeFile.WriteLine(weightLine);
                    }
                    writeFile.WriteLine("*");
                    //string biasLine = "";
                    for (int f = 0; f < net.layers[l].conv2DLayer.no_filters; f++)
                    {
                        writeFile.WriteLine(net.layers[l].biases.valuesConv[f]);
                    }
                    //biasLine = biasLine.Remove(biasLine.Length - 2);
                    // writeFile.WriteLine(biasLine);
                }
            }
            else if (net.layers[l].layer_type == Noedify.LayerType.FullyConnected | net.layers[l].layer_type == Noedify.LayerType.Output)
            {

                for (int j = 0; j < net.layers[l].layerSize; j++)
                {
                    string weightLine = "";
                    for (int i = 0; i < net.layers[l - 1].layerSize; i++)
                        weightLine += net.layers[l].weights.values[i, j] + ",";
                    weightLine = weightLine.Remove(weightLine.Length - 2);
                    writeFile.WriteLine(weightLine);
                }
                writeFile.WriteLine("*");
                for (int j = 0; j < net.layers[l].layerSize; j++)
                {
                    //string biasLine = "";
                    //biasLine += net.layers[l].biases.values[j] + ",";
                    //biasLine = biasLine.Remove(biasLine.Length - 1);
                    writeFile.WriteLine(net.layers[l].biases.values[j]);
                    //writeFile.WriteLine(biasLine);
                }
            }
            writeFile.WriteLine("***");
        }

        writeFile.Close();
    }

    static float[] ParseParameterLine(StreamReader file, int lineSize)
    {
        char[] separators = new char[lineSize];
        for (int k = 0; k < lineSize; k++)
            separators[k] = ',';
        string paramLine = file.ReadLine();
        //Debug.Log("new line (" + lineSize + " seps): " + paramLine);
        if (paramLine == "***" | paramLine == "*" | paramLine == "" | paramLine == " " | string.IsNullOrEmpty(paramLine))
        {
            return new float[0];
        }

        string[] paramStrings = paramLine.Split(separators);

        if (paramStrings.Length != lineSize)
            return new float[paramStrings.Length];

        float[] paramValues = new float[lineSize];
        for (int k = 0; k < lineSize; k++)
        {
            float.TryParse(paramStrings[k], out paramValues[k]);
        }
        return paramValues;
    }

    [System.Serializable]
    class ShortByteArray
    {
        public byte[] byte0;
        public byte[] byte1;

        public ShortByteArray(int size)
        {
            byte0 = new byte[size];
            byte1 = new byte[size];
        }

        public ShortByteArray()
        {
            byte0 = new byte[1];
            byte1 = new byte[1];
        }
    }

    static SingleByteArray CompressIntArray(int[] data, int sizeA)
    {
        SingleByteArray byteArray = new SingleByteArray(sizeA);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        byte[] bytes_uncmpr0 = new byte[sizeA];
        byte[] bytes_uncmpr1 = new byte[sizeA];
        byte[] bytes_uncmpr2 = new byte[sizeA];
        byte[] bytes_uncmpr3 = new byte[sizeA];

        for (int i = 0; i < sizeA; i++)
        {
            byte[] newBytes = System.BitConverter.GetBytes(data[i]);

            bytes_uncmpr0[i] = newBytes[0];
            bytes_uncmpr1[i] = newBytes[1];
            bytes_uncmpr2[i] = newBytes[2];
            bytes_uncmpr3[i] = newBytes[3];
        }

        using (DeflateStream dstream = new DeflateStream(output0, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr0, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output1, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr1, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output2, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr2, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output3, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr3, 0, data.Length);
        }

        byteArray.byte0 = output0.ToArray();
        byteArray.byte1 = output1.ToArray();
        byteArray.byte2 = output2.ToArray();
        byteArray.byte3 = output3.ToArray();

        return byteArray;
    }
    static SingleByteArray CompressIntArray(int[,] data, int sizeA, int sizeB, bool debug = false)
    {
        SingleByteArray byteArray = new SingleByteArray(sizeA * sizeB);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        byte[] bytes_uncmpr0 = new byte[sizeA * sizeB];
        byte[] bytes_uncmpr1 = new byte[sizeA * sizeB];
        byte[] bytes_uncmpr2 = new byte[sizeA * sizeB];
        byte[] bytes_uncmpr3 = new byte[sizeA * sizeB];

        for (int i = 0; i < sizeA; i++)
            for (int j = 0; j < sizeB; j++)
            {
                if (debug)
                    Debug.Log("Compress32BitArray<" + typeof(int) + "[,]>:  (i,j) = (" + i + "," + j + ")");
                byte[] newBytes = System.BitConverter.GetBytes(data[i, j]);
                bytes_uncmpr0[i * sizeB + j] = newBytes[0];
                bytes_uncmpr1[i * sizeB + j] = newBytes[1];
                bytes_uncmpr2[i * sizeB + j] = newBytes[2];
                bytes_uncmpr3[i * sizeB + j] = newBytes[3];
            }

        using (DeflateStream dstream = new DeflateStream(output0, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr0, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output1, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr1, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output2, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr2, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output3, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr3, 0, data.Length);
        }

        byteArray.byte0 = output0.ToArray();
        byteArray.byte1 = output1.ToArray();
        byteArray.byte2 = output2.ToArray();
        byteArray.byte3 = output3.ToArray();

        return byteArray;
    }

    public static SingleByteArray CompressFloatArray(float[] data, int sizeA)
    {
        SingleByteArray byteArray = new SingleByteArray(sizeA);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        byte[] bytes_uncmpr0 = new byte[sizeA];
        byte[] bytes_uncmpr1 = new byte[sizeA];
        byte[] bytes_uncmpr2 = new byte[sizeA];
        byte[] bytes_uncmpr3 = new byte[sizeA];

        for (int i = 0; i < sizeA; i++)
        {
            byte[] newBytes = System.BitConverter.GetBytes(data[i]);

            bytes_uncmpr0[i] = newBytes[0];
            bytes_uncmpr1[i] = newBytes[1];
            bytes_uncmpr2[i] = newBytes[2];
            bytes_uncmpr3[i] = newBytes[3];
        }

        using (DeflateStream dstream = new DeflateStream(output0, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr0, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output1, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr1, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output2, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr2, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output3, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr3, 0, data.Length);
        }

        byteArray.byte0 = output0.ToArray();
        byteArray.byte1 = output1.ToArray();
        byteArray.byte2 = output2.ToArray();
        byteArray.byte3 = output3.ToArray();

        return byteArray;
    }
    static SingleByteArray CompressFloatArray(float[,] data, int sizeA, int sizeB, bool debug = false)
    {
        SingleByteArray byteArray = new SingleByteArray(sizeA * sizeB);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        byte[] bytes_uncmpr0 = new byte[sizeA * sizeB];
        byte[] bytes_uncmpr1 = new byte[sizeA * sizeB];
        byte[] bytes_uncmpr2 = new byte[sizeA * sizeB];
        byte[] bytes_uncmpr3 = new byte[sizeA * sizeB];

        for (int i = 0; i < sizeA; i++)
            for (int j = 0; j < sizeB; j++)
            {
                if (debug)
                    Debug.Log("Compress32BitArray<" + typeof(int) + "[,]>:  (i,j) = (" + i + "," + j + ")");
                byte[] newBytes = System.BitConverter.GetBytes(data[i, j]);
                bytes_uncmpr0[i * sizeB + j] = newBytes[0];
                bytes_uncmpr1[i * sizeB + j] = newBytes[1];
                bytes_uncmpr2[i * sizeB + j] = newBytes[2];
                bytes_uncmpr3[i * sizeB + j] = newBytes[3];
            }

        using (DeflateStream dstream = new DeflateStream(output0, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr0, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output1, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr1, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output2, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr2, 0, data.Length);
        }
        using (DeflateStream dstream = new DeflateStream(output3, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(bytes_uncmpr3, 0, data.Length);
        }

        byteArray.byte0 = output0.ToArray();
        byteArray.byte1 = output1.ToArray();
        byteArray.byte2 = output2.ToArray();
        byteArray.byte3 = output3.ToArray();

        return byteArray;
    }

    static int[,] DecompressIntArray(SingleByteArray singleByteArray, int sizeA, int sizeB)
    {
        int[,] data = new int[sizeA, sizeB];

        MemoryStream input0 = new MemoryStream(singleByteArray.byte0);
        MemoryStream input1 = new MemoryStream(singleByteArray.byte1);
        MemoryStream input2 = new MemoryStream(singleByteArray.byte2);
        MemoryStream input3 = new MemoryStream(singleByteArray.byte3);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        using (DeflateStream dstream = new DeflateStream(input0, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output0);
        }
        using (DeflateStream dstream = new DeflateStream(input1, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output1);
        }
        using (DeflateStream dstream = new DeflateStream(input2, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output2);
        }
        using (DeflateStream dstream = new DeflateStream(input3, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output3);
        }

        byte[] bytes0 = output0.ToArray();
        byte[] bytes1 = output1.ToArray();
        byte[] bytes2 = output2.ToArray();
        byte[] bytes3 = output3.ToArray();

        for (int i = 0; i < sizeA; i++)
            for (int j = 0; j < sizeB; j++)
            {
                int byteIndex = i * sizeB + j;
                //print(byteIndex);
                byte[] byteArray = new byte[4] { bytes0[byteIndex], bytes1[byteIndex], bytes2[byteIndex], bytes3[byteIndex] };
                data[i, j] = System.BitConverter.ToInt32(byteArray, 0);
            }
        return data;
    }

    static int[] DecompressIntArray(SingleByteArray singleByteArray, int sizeA)
    {
        int[] data = new int[sizeA];

        MemoryStream input0 = new MemoryStream(singleByteArray.byte0);
        MemoryStream input1 = new MemoryStream(singleByteArray.byte1);
        MemoryStream input2 = new MemoryStream(singleByteArray.byte2);
        MemoryStream input3 = new MemoryStream(singleByteArray.byte3);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        using (DeflateStream dstream = new DeflateStream(input0, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output0);
        }
        using (DeflateStream dstream = new DeflateStream(input1, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output1);
        }
        using (DeflateStream dstream = new DeflateStream(input2, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output2);
        }
        using (DeflateStream dstream = new DeflateStream(input3, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output3);
        }

        byte[] bytes0 = output0.ToArray();
        byte[] bytes1 = output1.ToArray();
        byte[] bytes2 = output2.ToArray();
        byte[] bytes3 = output3.ToArray();

        for (int i = 0; i < sizeA; i++)
        {
            int byteIndex = i;
            //print(byteIndex);
            byte[] byteArray = new byte[4] { bytes0[byteIndex], bytes1[byteIndex], bytes2[byteIndex], bytes3[byteIndex] };
            data[i] = System.BitConverter.ToInt32(byteArray, 0);
        }
        return data;
    }

    static float[,] DecompressFloatArray(SingleByteArray singleByteArray, int sizeA, int sizeB)
    {
        float[,] data = new float[sizeA, sizeB];

        MemoryStream input0 = new MemoryStream(singleByteArray.byte0);
        MemoryStream input1 = new MemoryStream(singleByteArray.byte1);
        MemoryStream input2 = new MemoryStream(singleByteArray.byte2);
        MemoryStream input3 = new MemoryStream(singleByteArray.byte3);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();

        using (DeflateStream dstream = new DeflateStream(input0, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output0);
        }
        using (DeflateStream dstream = new DeflateStream(input1, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output1);
        }
        using (DeflateStream dstream = new DeflateStream(input2, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output2);
        }
        using (DeflateStream dstream = new DeflateStream(input3, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output3);
        }

        byte[] bytes0 = output0.ToArray();
        byte[] bytes1 = output1.ToArray();
        byte[] bytes2 = output2.ToArray();
        byte[] bytes3 = output3.ToArray();

        for (int i = 0; i < sizeA; i++)
            for (int j = 0; j < sizeB; j++)
            {
                int byteIndex = i * sizeB + j;
                //print(byteIndex);
                byte[] byteArray = new byte[4] { bytes0[byteIndex], bytes1[byteIndex], bytes2[byteIndex], bytes3[byteIndex] };
                data[i, j] = System.BitConverter.ToSingle(byteArray, 0);
            }
        return data;
    }

    public static float[] DecompressFloatArray(SingleByteArray singleByteArray, int sizeA, bool debug = false)
    {
        float[] data = new float[sizeA];
        if (debug) Debug.Log("creating streams");

        MemoryStream input0 = new MemoryStream(singleByteArray.byte0);
        MemoryStream input1 = new MemoryStream(singleByteArray.byte1);
        MemoryStream input2 = new MemoryStream(singleByteArray.byte2);
        MemoryStream input3 = new MemoryStream(singleByteArray.byte3);

        MemoryStream output0 = new MemoryStream();
        MemoryStream output1 = new MemoryStream();
        MemoryStream output2 = new MemoryStream();
        MemoryStream output3 = new MemoryStream();
        if (debug) Debug.Log("copying streams");
        using (DeflateStream dstream = new DeflateStream(input0, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output0);
        }
        if (debug) Debug.Log("copying stream 1");
        using (DeflateStream dstream = new DeflateStream(input1, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output1);
        }
        using (DeflateStream dstream = new DeflateStream(input2, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output2);
        }
        using (DeflateStream dstream = new DeflateStream(input3, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output3);
        }
        if (debug) Debug.Log("outputting values");

        byte[] bytes0 = output0.ToArray();
        byte[] bytes1 = output1.ToArray();
        byte[] bytes2 = output2.ToArray();
        byte[] bytes3 = output3.ToArray();

        for (int i = 0; i < sizeA; i++)
        {
            int byteIndex = i;
            //print(byteIndex);
            byte[] byteArray = new byte[4] { bytes0[byteIndex], bytes1[byteIndex], bytes2[byteIndex], bytes3[byteIndex] };
            data[i] = System.BitConverter.ToSingle(byteArray, 0);
        }
        return data;
    }

    private static byte[] ConvertToBytes<T>(T[,] inputData, int sizeA, int sizeB)
    {
        byte[] outputBytes = new byte[sizeA * sizeB];
        for (int i = 0; i < sizeA; i++)
            for (int j = 0; j < sizeB; j++)
                outputBytes[i * sizeB + j] = System.Convert.ToByte(inputData[i, j]);
        return outputBytes;
    }

    private static byte[] ConvertToBytes<T>(T[] inputData, int sizeA)
    {
        byte[] outputBytes = new byte[sizeA];
        for (int i = 0; i < sizeA; i++)
        {
            outputBytes[i] = System.Convert.ToByte(inputData[i]);
        }
        return outputBytes;
    }

    private static int[,] ConvertIntFromBytes(byte[] dataBytes, int sizeA, int sizeB)
    {
        int[,] outputArray = new int[sizeA, sizeB];

        for (int i = 0; i < sizeA; i++)
            for (int j = 0; j < sizeB; j++)
            {
                outputArray[i, j] = System.Convert.ToInt32(dataBytes[i * sizeB + j]);

            }
        return outputArray;
    }

    private static int[] ConvertIntFromBytes(byte[] dataBytes, int sizeA)
    {
        int[] outputArray = new int[sizeA];
        for (int i = 0; i < sizeA; i++)
            outputArray[i] = System.Convert.ToInt32(dataBytes[i]);
        return outputArray;
    }

    private static byte[] CompressData(byte[] data)
    {
        MemoryStream output = new MemoryStream();
        using (DeflateStream dstream = new DeflateStream(output, System.IO.Compression.CompressionLevel.Optimal))
        {
            dstream.Write(data, 0, data.Length);
        }
        return output.ToArray();
    }

    private static byte[] DecompressData(byte[] data)
    {
        MemoryStream input = new MemoryStream(data);
        MemoryStream output = new MemoryStream();
        using (DeflateStream dstream = new DeflateStream(input, System.IO.Compression.CompressionMode.Decompress))
        {
            dstream.CopyTo(output);
        }
        return output.ToArray();
    }


}
