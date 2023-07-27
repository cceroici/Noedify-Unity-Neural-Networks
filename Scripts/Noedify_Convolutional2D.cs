//#define CONNMASK
#define COMPRESSCONNECTIONS

using UnityEngine;
using System.Collections.Generic;

public class Noedify_Convolutional2D : MonoBehaviour
{
    [System.Serializable]
    public class Convolutional2DLayer
    {
        public int[] stride;
        public int no_filters;
        public int[] filterSize;
        public int[] padding;
        public int[,] connections;
        public int[,] connectionsInFilter;
        public bool[,,,] connections2D;
        public int[,] connectionMask;
        public int[] filterTrack;
        public int[] channelTrack;
        public int[] nodeTrack;
        public int N_weights_per_filter;
        public int N_connections_per_node;

        public Convolutional2DLayer()
        {
            stride = new int[2];
            padding = new int[2];
            filterSize = new int[2];
        }

        public void BuildConnections(Noedify.Layer previousLayer, Noedify.Layer layer)
        {
            //print("Building convolutional connections");
            bool prevLyrInputType = !(new List<Noedify.LayerType>() { Noedify.LayerType.Convolutional2D, Noedify.LayerType.Pool2D, Noedify.LayerType.BatchNorm2D }.Contains(previousLayer.layer_type));

            //if (prevLyrInputType)
            //    N_weights_per_filter = filterSize[0] * filterSize[1];
            //else 
            ///////////////////////////////////////////////////////
             N_weights_per_filter = filterSize[0] * filterSize[1] * previousLayer.conv2DLayer.no_filters;

#if CONNMASK
            connectionMask = new int[previousLayer.layerSize, layer.layerSize];
#endif
            filterTrack = new int[layer.layerSize];
            channelTrack = new int[layer.layerSize];
            nodeTrack = new int[layer.layerSize];

            int nodesPerFilter = layer.layerSize2D[0] * layer.layerSize2D[1];

            int currentChannel = 0;
            int currentFilter = 0;
            int currentNodeInChannel = 0;

            N_connections_per_node = filterSize[0] * filterSize[1]; // Number of connections per node

            if (prevLyrInputType)
                N_connections_per_node *= previousLayer.in_channels;
            else
                N_connections_per_node *= previousLayer.conv2DLayer.no_filters;

            for (int j = 0; j < layer.layerSize; j++)
            {
                filterTrack[j] = currentFilter;
                channelTrack[j] = currentChannel;
                nodeTrack[j] = currentNodeInChannel;
                currentNodeInChannel++;
                if (currentNodeInChannel == nodesPerFilter)
                {
                    currentNodeInChannel = 0;
                    currentFilter++;
                    if (currentFilter == layer.conv2DLayer.no_filters)
                    {
                        currentFilter = 0;
                        currentChannel++;
                    }
                }
            }
#if COMPRESSCONNECTIONS
            connections = new int[layer.layerSize2D[0] * layer.layerSize2D[1], N_connections_per_node];
            connectionsInFilter = new int[layer.layerSize2D[0] * layer.layerSize2D[1], N_connections_per_node];
#else
            connections = new int[layer.layerSize, N_connections_per_node];
            connectionsInFilter = new int[layer.layerSize, N_connections_per_node];
#endif
            int input_channels = 1;
            if (prevLyrInputType)
                input_channels = previousLayer.in_channels;
            else
                input_channels = previousLayer.conv2DLayer.no_filters;
#if COMPRESSCONNECTIONS
            for (int j = 0; j < layer.layerSize2D[0] * layer.layerSize2D[1]; j++)
# else
            for (int j = 0; j < layer.layerSize; j++)
#endif
                for (int fi = 0; fi < input_channels * filterSize[0] * filterSize[1]; fi++)
                {
                    connections[j, fi] = -1;
                    connectionsInFilter[j, fi] = -1;
                }
            int total_no_connections = 0;

            for (int c = 0; c < input_channels; c++)
            {
                int[,] ConnectionMeshGrid = new int[previousLayer.layerSize2D[0], previousLayer.layerSize2D[1]];


                for (int i = 0; i < previousLayer.layerSize2D[0]; i++)
                {
                    //string outString = "ConnectionMeshgrid: ";
                    for (int j = 0; j < previousLayer.layerSize2D[1]; j++)
                    {
                        ConnectionMeshGrid[i, j] = c * previousLayer.layerSize2D[1] * previousLayer.layerSize2D[0] + i * previousLayer.layerSize2D[1] + j;
                        //outString += ConnectionMeshGrid[i, j] + " ";
                    }
                    //print(outString);
                }
#if COMPRESSCONNECTIONS
                for (int f = 0; f < 1; f++)
#else
                for (int f = 0; f < no_filters; f++)
#endif
                {
                    int index = c * filterSize[0] * filterSize[1];
                    int w_pos = -padding[1];
                    int h_pos = -padding[0];
                    int[] Steps = new int[2];
                    /*
                    Steps[0] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[0] - (float)filterSize[0]) / (float)stride[0] + 1f);
                    Steps[1] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[1] - (float)filterSize[1]) / (float)stride[1] + 1f);
                    print("Input dimensions: (" + previousLayer.layerSize2D[0] + "," + previousLayer.layerSize2D[1] + ")");
                    print("Layer dimensions: (" + layer.layerSize2D[0] + "," + layer.layerSize2D[1] + ")");
                    //print("Filter dimensions: (" + filterSize[0] + "," + filterSize[1] + ")");
                    print("Steps: (" + Steps[0] + "," + Steps[1] + ")");
                    */
                    Steps = layer.layerSize2D;

                    for (int d = 0; d < Steps[0]; d++)
                    {
                        for (int g = 0; g < Steps[1]; g++)
                        {
                            //string printOutput = "connection map c=" + c + ", w=" + d + ", h=" + g + ": ";
                            for (int h = h_pos; h < (h_pos + filterSize[0]); h++)
                            {
                                for (int w = w_pos; w < (w_pos + filterSize[1]); w++)
                                {
                                    if ((h < previousLayer.layerSize2D[0]) & (w < previousLayer.layerSize2D[1]) & w >= 0 & h >= 0)
                                    {
                                        //if (layer.layer_no==3) print("Failed conn assignment at: c = " + c + ", f = " + f + ", d = " + d + ", g = " + g + ", sum: " + (f * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + index);
                                        connections[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] = ConnectionMeshGrid[h, w];
                                        connectionsInFilter[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] = c * filterSize[0] * filterSize[1] + (h - h_pos) * filterSize[1] + (w - w_pos);
                                        //printOutput += "(" + (f * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + "(f" + connectionsInFilter[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] + ") ";
                                        //printOutput += connectionsInFilter[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + " ";
                                        index++;
                                        total_no_connections++;
                                    }
                                }
                            }
                            // print(printOutput);

                            w_pos += stride[1];
                            index = c * filterSize[0] * filterSize[1];
                        }
                        h_pos += stride[0];
                        w_pos = -padding[1];
                    }
                }

            }
#if CONNMASK
            for (int i = 0; i < previousLayer.layerSize; i++)
                for (int j = 0; j < layer.layerSize; j++)
                    connectionMask[i, j] = -1;

            for (int j = 0; j < layer.layerSize; j++)
                for (int m = 0; m < N_connections_per_node; m++)
                    if (connections[j, m] >= 0)
                        connectionMask[connections[j, m], j] = connectionsInFilter[j, m];
#endif
        }

        public void BuildConnectionsPool2D(Noedify.Layer previousLayer, Noedify.Layer layer)
        {
            //print("starting pooling map");
            connectionMask = new int[previousLayer.layerSize, layer.layerSize];
            nodeTrack = new int[layer.layerSize];
            filterTrack = new int[layer.layerSize];
            no_filters = previousLayer.conv2DLayer.no_filters;

            int currentNode = 0;
            int currentFilter = 0;
            int nodesPerFilter = layer.layerSize2D[0] * layer.layerSize2D[1];

            N_connections_per_node = filterSize[0] * filterSize[1]; // Number of connections per node

            for (int j = 0; j < layer.layerSize; j++)
            {
                nodeTrack[j] = currentNode++;
                if (currentNode == nodesPerFilter)
                {
                    filterTrack[j] = currentFilter;
                    currentFilter++;
                }
            }

            connections = new int[layer.layerSize, N_connections_per_node];
            connectionsInFilter = new int[layer.layerSize, N_connections_per_node];

            for (int i = 0; i < layer.layerSize; i++)
                for (int j = 0; j < N_connections_per_node; j++)
                {
                    connections[i, j] = -1;
                    connectionsInFilter[i, j] = -1;
                }
            int total_no_connections = 0;

            for (int c = 0; c < previousLayer.conv2DLayer.no_filters; c++)
            {
                int[,] ConnectionMeshGrid = new int[previousLayer.layerSize2D[0], previousLayer.layerSize2D[1]];

                for (int i = 0; i < previousLayer.layerSize2D[0]; i++)
                {
                    //string outString = "ConnectionMeshgrid: ";
                    for (int j = 0; j < previousLayer.layerSize2D[1]; j++)
                    {
                        ConnectionMeshGrid[i, j] = c * previousLayer.layerSize2D[1] * previousLayer.layerSize2D[0] + i * previousLayer.layerSize2D[1] + j;
                        //outString += ConnectionMeshGrid[i, j] + " ";
                    }
                    //print(outString);
                }


                int index = 0;
                int w_pos = -padding[1];
                int h_pos = -padding[0];
                int[] Steps = new int[2];
                Steps[0] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[0] - (float)filterSize[0]) / (float)stride[0] + 1f);
                Steps[1] = Mathf.CeilToInt(((float)previousLayer.layerSize2D[1] - (float)filterSize[1]) / (float)stride[1] + 1f);
                //print("Input dimensions: (" + previousLayer.layerSize2D[0] + "," + previousLayer.layerSize2D[1] + ")");
                //print("Filter dimensions: (" + filterSize[0] + "," + filterSize[1] + ")");
                //print("Steps: (" + Steps[0] + "," + Steps[1] + ")");

                for (int d = 0; d < Steps[0]; d++)
                {
                    for (int g = 0; g < Steps[1]; g++)
                    {
                        //string printOutput = "connection map c=" + c + ", w=" + d + ", h=" + g + ": ";
                        for (int h = h_pos; h < (h_pos + filterSize[0]); h++)
                        {
                            for (int w = w_pos; w < (w_pos + filterSize[1]); w++)
                            {
                                if ((h < previousLayer.layerSize2D[0]) & (w < previousLayer.layerSize2D[1]) & w >= 0 & h >= 0)
                                {
                                    //print("Failed conn assignment at: c = " + c + ", (d,g) = (" + d + "," + g + "), (w,h) = (" + w + "," + h + "), connections[nodeIndex = " + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + (index));
                                    connections[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] = ConnectionMeshGrid[h, w];
                                    connectionsInFilter[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] = c * filterSize[0] * filterSize[1] + (h - h_pos) * filterSize[1] + (w - w_pos);
                                    total_no_connections++;
                                    index++;
                                    //printOutput += "(" + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index - 1].ToString() + " ";
                                    //printOutput += connectionsInFilter[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index - 1].ToString() + " ";

                                }
                            }
                        }
                        //print(printOutput);

                        w_pos += stride[1];
                        index = 0;
                    }
                    h_pos += stride[0];
                    w_pos = -padding[1];
                }


            }

            for (int i = 0; i < previousLayer.layerSize; i++)
                for (int j = 0; j < layer.layerSize; j++)
                    connectionMask[i, j] = -1;
            for (int j = 0; j < layer.layerSize; j++)
                for (int m = 0; m < N_connections_per_node; m++)
                    if (connections[j, m] >= 0)
                        connectionMask[connections[j, m], j] = connectionsInFilter[j, m];

        }

        public void BuildConnectionsTransConv2D(Noedify.Layer previousLayer, Noedify.Layer layer)
        {
            //print("building connections for layer: " + layer.name);
            //connectionMask = new int[previousLayer.layerSize, layer.layerSize];

            filterTrack = new int[layer.layerSize];
            channelTrack = new int[layer.layerSize];
            nodeTrack = new int[layer.layerSize];

            int nodesPerFilter = layer.layerSize2D[0] * layer.layerSize2D[1];

            int currentChannel = 0;
            int currentFilter = 0;
            int currentNodeInChannel = 0;
            for (int j = 0; j < layer.layerSize; j++)
            {
                filterTrack[j] = currentFilter;
                channelTrack[j] = currentChannel;
                nodeTrack[j] = currentNodeInChannel;
                currentNodeInChannel++;
                if (currentNodeInChannel == nodesPerFilter)
                {
                    currentNodeInChannel = 0;
                    currentFilter++;
                    if (currentFilter == layer.conv2DLayer.no_filters)
                    {
                        currentFilter = 0;
                        currentChannel++;
                    }
                }
            }

            switch (previousLayer.layer_type)
            {
                case (Noedify.LayerType.FullyConnected):
                case (Noedify.LayerType.Input):
                case (Noedify.LayerType.ActivationFunction):
                    {
#if CONNMASK
                        connectionMask = new int[previousLayer.layerSize, layer.layerSize];
                        N_weights_per_filter = filterSize[0] * filterSize[1];
                        for (int i = 0; i < previousLayer.layerSize; i++)
                            for (int j = 0; j < layer.layerSize; j++)
                                connectionMask[i, j] = j * N_weights_per_filter + i;
#endif
                        break;
                    }
                case (Noedify.LayerType.Convolutional2D):
                case (Noedify.LayerType.Input2D):
                case (Noedify.LayerType.Pool2D):
                case (Noedify.LayerType.BatchNorm2D):
                case (Noedify.LayerType.TranspConvolutional2D):
                    {
                        N_weights_per_filter = filterSize[0] * filterSize[1] * previousLayer.conv2DLayer.no_filters;
                        N_connections_per_node = filterSize[0] * filterSize[1] * no_filters; // Number of connections per node
#if COMPRESSCONNECTIONS
                        connections = new int[previousLayer.layerSize2D[0] * previousLayer.layerSize2D[1], N_connections_per_node];
                        connectionsInFilter = new int[previousLayer.layerSize2D[0] * previousLayer.layerSize2D[1], N_connections_per_node];
#else
                        connections = new int[previousLayer.layerSize, N_connections_per_node];
                        connectionsInFilter = new int[previousLayer.layerSize, N_connections_per_node];
#endif

#if COMPRESSCONNECTIONS
                        for (int i = 0; i < previousLayer.layerSize2D[0] * previousLayer.layerSize2D[1]; i++)
#else
                        for (int i = 0; i < previousLayer.layerSize; i++)
#endif
                            for (int j = 0; j < no_filters * filterSize[0] * filterSize[1]; j++)
                            {
                                connections[i, j] = -1;
                                connectionsInFilter[i, j] = -1;
                            }
                        int total_no_connections = 0;

#if COMPRESSCONNECTIONS
                        for (int c = 0; c < 1; c++)

#else
                        for (int c = 0; c < previousLayer.conv2DLayer.no_filters; c++)
#endif
                        {


                            for (int f = 0; f < no_filters; f++)
                            {
                                int index = f * filterSize[0] * filterSize[1];
                                int w_pos = -padding[1];
                                int h_pos = -padding[0];
                                int[] Steps = new int[2];

                                int[,] ConnectionMeshGrid = new int[layer.layerSize2D[0], layer.layerSize2D[1]];

                                for (int i = 0; i < layer.layerSize2D[0]; i++)
                                {
                                    //string outString = "ConnectionMeshgrid: ";
                                    for (int j = 0; j < layer.layerSize2D[1]; j++)
                                    {
                                        ConnectionMeshGrid[i, j] = f * layer.layerSize2D[1] * layer.layerSize2D[0] + i * layer.layerSize2D[1] + j;
                                        //outString += ConnectionMeshGrid[i, j] + " ";
                                    }
                                    //print(outString);
                                }

                                Steps = previousLayer.layerSize2D;

                                for (int d = 0; d < Steps[0]; d++) // previous node (y)
                                {
                                    for (int g = 0; g < Steps[1]; g++) // previous node (x) (each incremenet applies a stride to pos)
                                    {
                                        //string printOutput = "connection map c=" + c + ", w=" + d + ", h=" + g + ": ";
                                        for (int h = h_pos; h < (h_pos + filterSize[0]); h++)
                                        {
                                            for (int w = w_pos; w < (w_pos + filterSize[1]); w++)
                                            {
                                                if ((h < layer.layerSize2D[0]) & (w < layer.layerSize2D[1]) & w >= 0 & h >= 0)
                                                {
                                                    //print("(w,h) = (" + w + "," + h + ") , f=" + f + ", c=" + c);
                                                    //print("prev node: " + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + index + ", f=" + f);
                                                    connections[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] = ConnectionMeshGrid[h, w];
                                                    connectionsInFilter[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] =  c * filterSize[0] * filterSize[1] + (h - h_pos) * filterSize[1] + (w - w_pos);
                                                    //printOutput += "(" + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + "(f" + connectionsInFilter[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] + ") ";
                                                    index++;
                                                    total_no_connections++;
                                                }
                                            }
                                        }
                                        //print(printOutput);

                                        w_pos += stride[1];
                                        index = f * filterSize[0] * filterSize[1];
                                    }
                                    h_pos += stride[0];
                                    w_pos = -padding[1];
                                }
                            }

                            int[,] connections_rev;
                            int[,] connectionsInFilter_rev;

                            int N_connections_per_node_j = Mathf.CeilToInt((float)filterSize[0] / (float)stride[0]) * Mathf.CeilToInt((float)filterSize[1] / (float)stride[1]);
                            connections_rev = new int[layer.layerSize2D[0] * layer.layerSize2D[1], N_connections_per_node_j];
                            connectionsInFilter_rev = new int[layer.layerSize2D[0] * layer.layerSize2D[1], N_connections_per_node_j];
                            for (int j=0; j < layer.layerSize2D[0] * layer.layerSize2D[1]; j++)
                                for (int conn=0; conn < N_connections_per_node_j; conn++)
                                {
                                    connections_rev[j, conn] = -1;
                                    connectionsInFilter_rev[j, conn] = -1;
                                }
                            //print("Conns per node: " + N_connections_per_node + ", (rev) " + N_connections_per_node_j);
                            int[] connN_j = new int[layer.layerSize2D[0] * layer.layerSize2D[1]];
                            for (int i=0; i < previousLayer.layerSize2D[0]* previousLayer.layerSize2D[1]; i++)
                            {
                                //print("--- source node i=" + i);
                                for (int conn=0; conn < N_connections_per_node; conn++)
                                {
                                    int connected_node_j = connections[i, conn];
                                    int cif_j = connectionsInFilter[i, conn];
                                    if ((connected_node_j > -1) & (connected_node_j < (layer.layerSize2D[0] * layer.layerSize2D[1])))
                                    {
                                        //print("connected_node_j: " + connected_node_j + ", cif_j: " + cif_j);
                                        //print("connections[" + connected_node_j + ", " + N_connections_per_node_j + " - " + cif_j + "(cif) - 1]" + " = " + i);
                                        //print("connections[" + connected_node_j + ", " + (N_connections_per_node_j - cif_j - 1) + "] = " + i);
                                        //connections_rev[connected_node_j, N_connections_per_node_j - cif_j - 1] = i;
                                        //connectionsInFilter_rev[connected_node_j, N_connections_per_node_j - cif_j - 1] = cif_j;

                                        connections_rev[connected_node_j, connN_j[connected_node_j]] = i;
                                        connectionsInFilter_rev[connected_node_j, connN_j[connected_node_j]] = cif_j;
                                        connN_j[connected_node_j]++;
                                    }
                                    //connections_rev[]
                                }
                            }
                            /*
                            print("-------- Normal:");
                            print("Connections per node: " + N_connections_per_node);
                            print("Connections: ");
                            for (int i = 0; i < previousLayer.layerSize2D[0] * previousLayer.layerSize2D[1]; i++)
                            {
                                string conString = "conn [" + i + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node; conn++)
                                {
                                    conString += connections[i, conn] + ", ";
                                }
                                print(conString);
                            }
                            print("CIF: ");
                            for (int i = 0; i < previousLayer.layerSize2D[0] * previousLayer.layerSize2D[1]; i++)
                            {
                                string cifString = "cif [" + i + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node; conn++)
                                {
                                    cifString += connectionsInFilter[i, conn] + ", ";
                                }
                                print(cifString);
                            }
                            print("-------- Reversed:");
                            print("Connections per node: " + N_connections_per_node_j);
                            print("Connections: ");
                            for (int j = 0; j < layer.layerSize2D[0] * layer.layerSize2D[1]; j++)
                            {
                                string conString = "conn [" + j + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node_j; conn++)
                                {
                                    conString += connections_rev[j, conn] + ", ";
                                }
                                print(conString);
                            }
                            print("CIF: ");
                            for (int j = 0; j < layer.layerSize2D[0] * layer.layerSize2D[1]; j++)
                            {
                                string cifString = "cif [" + j + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node_j; conn++)
                                {
                                    cifString += connectionsInFilter_rev[j, conn] + ", ";
                                }
                                print(cifString);
                            }

                            */
                            connections = connections_rev;
                            connectionsInFilter = connectionsInFilter_rev;
                            N_connections_per_node = N_connections_per_node_j;
                        }
                        
                        break;
                    }
            }

        }


        public void BuildTrackers2D(Noedify.Layer layer)
        {
            filterTrack = new int[layer.layerSize];
            channelTrack = new int[layer.layerSize];
            nodeTrack = new int[layer.layerSize];

            int nodesPerFilter = layer.layerSize2D[0] * layer.layerSize2D[1];

            int currentChannel = 0;
            int currentFilter = 0;
            int currentNodeInChannel = 0;

            for (int j = 0; j < layer.layerSize; j++)
            {
                filterTrack[j] = currentFilter;
                channelTrack[j] = currentChannel;
                nodeTrack[j] = currentNodeInChannel;
                currentNodeInChannel++;
                if (currentNodeInChannel == nodesPerFilter)
                {
                    currentNodeInChannel = 0;
                    currentFilter++;
                    if (currentFilter == no_filters)
                    {
                        currentFilter = 0;
                        currentChannel++;
                    }
                }
            }
            /*
            string outString = "layer " + layer.layer_no + " ";
            for (int i = 0; i < layer.layerSize; i++)
                outString += "(node=" + i + ")(nodeIC=" + nodeTrack[i] + ")(filterTrack=" + filterTrack[i] + ")(chTrack=" + channelTrack[i] + ") , ";
            print(outString);
            */
        }
    }

}
