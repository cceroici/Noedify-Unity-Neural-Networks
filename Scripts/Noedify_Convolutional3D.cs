//#define CONNMASK
#define COMPRESSCONNECTIONS

using UnityEngine;
using System.Collections.Generic;

public class Noedify_Convolutional3D : MonoBehaviour
{
    [System.Serializable]
    public class Convolutional3DLayer
    {
        public int[] stride;
        public int no_filters;
        public int[] filterSize;
        public int[] padding;
        public int[,] connections;
        public int[,] connectionsInFilter;
        public bool[,,,] connections3D;
        public int[,] connectionMask;
        public int[] filterTrack;
        public int[] channelTrack;
        public int[] nodeTrack;
        public int N_weights_per_filter;
        public int N_connections_per_node;

        public Convolutional3DLayer()
        {
            stride = new int[3];
            padding = new int[3];
            filterSize = new int[3];
        }

        public void BuildConnections(Noedify.Layer previousLayer, Noedify.Layer layer)
        {
            //print("Building convolutional connections");
            bool prevLyrInputType = !(new List<Noedify.LayerType>() { Noedify.LayerType.Convolutional3D, Noedify.LayerType.BatchNorm3D }.Contains(previousLayer.layer_type));

            N_weights_per_filter = filterSize[0] * filterSize[1] * filterSize[2] * previousLayer.conv3DLayer.no_filters;

            filterTrack = new int[layer.layerSize];
            channelTrack = new int[layer.layerSize];
            nodeTrack = new int[layer.layerSize];

            int nodesPerFilter = layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2] ;

            int currentChannel = 0;
            int currentFilter = 0;
            int currentNodeInChannel = 0;

            N_connections_per_node = filterSize[0] * filterSize[1] * filterSize[2]; // Number of connections per node

            if (prevLyrInputType)
                N_connections_per_node *= previousLayer.in_channels;
            else
                N_connections_per_node *= previousLayer.conv3DLayer.no_filters;

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
                    if (currentFilter == layer.conv3DLayer.no_filters)
                    {
                        currentFilter = 0;
                        currentChannel++;
                    }
                }
            }
            connections = new int[layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2], N_connections_per_node];
            connectionsInFilter = new int[layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2], N_connections_per_node];

            int input_channels = 1;
            if (prevLyrInputType)
                input_channels = previousLayer.in_channels;
            else
                input_channels = previousLayer.conv3DLayer.no_filters;
            for (int j = 0; j < layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]; j++)

                for (int fi = 0; fi < input_channels * filterSize[0] * filterSize[1]; fi++)
                {
                    connections[j, fi] = -1;
                    connectionsInFilter[j, fi] = -1;
                }
            int total_no_connections = 0;

            for (int c = 0; c < input_channels; c++)
            {
                int[,,] ConnectionMeshGrid = new int[previousLayer.layerSize3D[0], previousLayer.layerSize3D[1], previousLayer.layerSize3D[2]];

                /*
                for (int i = 0; i < previousLayer.layerSize3D[0]; i++)
                {
                    string outString = "ConnectionMeshgrid: ";
                    for (int j = 0; j < previousLayer.layerSize3D[1]; j++)
                    {
                        for (int k = 0; k < previousLayer.layerSize3D[2]; k++)
                        {
                            ConnectionMeshGrid[i, j, k] = c * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[0] * previousLayer.layerSize3D[2] +
                                i * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2] + j * previousLayer.layerSize3D[2] + k;
                            outString += ConnectionMeshGrid[i, j, k] + " ";
                        }
                        outString += ", ";
                    }
                    print(outString);
                }
                */
                
                for (int k = 0; k < previousLayer.layerSize3D[2]; k++)
                {
                    //string outString = "ConnectionMeshgrid: ";
                    for (int j = 0; j < previousLayer.layerSize3D[1]; j++)
                    {
                        for (int i = 0; i < previousLayer.layerSize3D[0]; i++)
                        {
                            ConnectionMeshGrid[i, j, k] = c * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[0] * previousLayer.layerSize3D[2] +
                                k * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[0] + j * previousLayer.layerSize3D[0] + i;
                            //outString += ConnectionMeshGrid[i, j, k] + " ";
                        }
                        //outString += ", ";
                    }
                    //print(outString);
                }
                #region old connection order
                /*
                for (int f = 0; f < 1; f++)
                {
                    int index = c * filterSize[0] * filterSize[1] * filterSize[2];
                    int u_pos = -padding[2];
                    int w_pos = -padding[1];
                    int h_pos = -padding[0];
                    int[] Steps = new int[3];

                    Steps = layer.layerSize3D;

                    for (int d = 0; d < Steps[0]; d++)
                    {
                        for (int g = 0; g < Steps[1]; g++)
                        {
                            for (int p = 0; p < Steps[2]; p++)
                            {
                                //string printOutput = "connection map c=" + c + ", w=" + d + ", h=" + g + ": ";
                                for (int h = h_pos; h < (h_pos + filterSize[0]); h++)
                                {
                                    for (int w = w_pos; w < (w_pos + filterSize[1]); w++)
                                    {
                                        for (int u = u_pos; u < (u_pos + filterSize[2]); u++)
                                        {
                                            if ((h < previousLayer.layerSize3D[0]) & (w < previousLayer.layerSize3D[1]) & (u < previousLayer.layerSize3D[2]) & w >= 0 & h >= 0 & u >= 0)
                                            {
                                                //if (layer.layer_no==3) print("Failed conn assignment at: c = " + c + ", f = " + f + ", d = " + d + ", g = " + g + ", sum: " + (f * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + index);
                                                connections[f * (Steps[0] * Steps[1] * Steps[2]) + d * Steps[1] * Steps[2] + g * Steps[2] + p, index] = ConnectionMeshGrid[h, w, u];
                                                connectionsInFilter[f * (Steps[0] * Steps[1] * Steps[2]) + d * Steps[1] * Steps[2] + g * Steps[2] + p, index] = c * filterSize[0] * filterSize[1] * filterSize[2] +
                                                    (h - h_pos) * filterSize[1] * filterSize[2] + (w - w_pos) * filterSize[2] + (u-u_pos);
                                                //printOutput += "(" + (f * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + "(f" + connectionsInFilter[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] + ") ";
                                                //printOutput += connectionsInFilter[f * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + " ";
                                                index++;
                                                total_no_connections++;
                                            }
                                        }
                                    }
                                }
                                u_pos += stride[2];
                                index = c * filterSize[0] * filterSize[1] * filterSize[2];
                                //print(printOutput);
                            }
                            
                            w_pos += stride[1];
                            u_pos = -padding[2];
                            
                        }
                        h_pos += stride[0];
                        w_pos = -padding[1];
                    }
                }
                */
                #endregion
                for (int f = 0; f < 1; f++)
                {
                    int index = c * filterSize[0] * filterSize[1] * filterSize[2];
                    int u_pos = -padding[2];
                    int w_pos = -padding[0];
                    int h_pos = -padding[1];
                    int[] Steps = new int[3];

                    Steps = layer.layerSize3D;

                    for (int p = 0; p < Steps[2]; p++)
                    {
                        for (int g = 0; g < Steps[1]; g++)
                        {
                            for (int d = 0; d < Steps[0]; d++)
                            {
                                int node_index = f * (Steps[0] * Steps[1] * Steps[2]) + p * Steps[0] * Steps[1] + g * Steps[0] + d;
                                //string printOutput = "connection map (node=" + node_index + ") c=" + c + ", w=" + d + ", h=" + g + ", d=" + p + ": ";
                                for (int u = u_pos; u < (u_pos + filterSize[2]); u++)
                                {
                                    for (int h = h_pos; h < (h_pos + filterSize[1]); h++)
                                    {
                                        for (int w = w_pos; w < (w_pos + filterSize[0]); w++)
                                        {
                                            if ((h < previousLayer.layerSize3D[1]) & (w < previousLayer.layerSize3D[0]) & (u < previousLayer.layerSize3D[2]) & w >= 0 & h >= 0 & u >= 0)
                                            {
                                                //if (layer.layer_no==3) print("Failed conn assignment at: c = " + c + ", f = " + f + ", d = " + d + ", g = " + g + ", sum: " + (f * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + index);
                                                connections[node_index, index] = ConnectionMeshGrid[w, h, u];
                                                connectionsInFilter[node_index, index] = c * filterSize[0] * filterSize[1] * filterSize[2] +
                                                    (u - u_pos) * filterSize[0] * filterSize[1] + (h - h_pos) * filterSize[0] + (w - w_pos);
                                                //printOutput += "(" + (f * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[f * (Steps[0] * Steps[1] * Steps[2]) + p * Steps[0] * Steps[1] + g * Steps[0] + d, index].ToString() + "(f" + connectionsInFilter[f * (Steps[0] * Steps[1] * Steps[2]) + p * Steps[0] * Steps[1] + g * Steps[2] + d, index] + ") ";
                                                //printOutput += connectionsInFilter[f * (Steps[0] * Steps[1] * Steps[2]) + p * Steps[0] * Steps[1] + g * Steps[2] + d, index].ToString() + " ";
                                                index++;
                                                total_no_connections++;
                                            }
                                        }
                                    }
                                }
                                w_pos += stride[0];
                                index = c * filterSize[0] * filterSize[1] * filterSize[2];
                                //print(printOutput);
                            }

                            h_pos += stride[1];
                            w_pos = -padding[0];

                        }
                        u_pos += stride[2];
                        h_pos = -padding[1];
                    }
                }
            }

        }


        public void BuildConnectionsTransConv3D(Noedify.Layer previousLayer, Noedify.Layer layer)
        {
            //print("building connections for layer: " + layer.name);
            //connectionMask = new int[previousLayer.layerSize, layer.layerSize];

            filterTrack = new int[layer.layerSize];
            channelTrack = new int[layer.layerSize];
            nodeTrack = new int[layer.layerSize];

            int nodesPerFilter = layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2];

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
                    if (currentFilter == layer.conv3DLayer.no_filters)
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
                        break;
                    }
                case (Noedify.LayerType.Convolutional3D):
                case (Noedify.LayerType.Input3D):
                case (Noedify.LayerType.BatchNorm3D):
                case (Noedify.LayerType.TranspConvolutional3D):
                    {
                        N_weights_per_filter = filterSize[0] * filterSize[1] * filterSize[2] * previousLayer.conv3DLayer.no_filters;
                        N_connections_per_node = filterSize[0] * filterSize[1] * filterSize[2] * no_filters; // Number of connections per node
                        connections = new int[previousLayer.layerSize3D[0] * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2], N_connections_per_node];
                        connectionsInFilter = new int[previousLayer.layerSize3D[0] * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2], N_connections_per_node];


                        for (int i = 0; i < previousLayer.layerSize3D[0] * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2]; i++)
                            for (int j = 0; j < no_filters * filterSize[0] * filterSize[1] * filterSize[2]; j++)
                            {
                                connections[i, j] = -1;
                                connectionsInFilter[i, j] = -1;
                            }
                        int total_no_connections = 0;

                        for (int c = 0; c < 1; c++)
                        {
                            for (int f = 0; f < no_filters; f++)
                            {
                                /*
                                int index = f * filterSize[0] * filterSize[1] * filterSize[2];
                                int u_pos = -padding[2];
                                int w_pos = -padding[1];
                                int h_pos = -padding[0];
                                int[] Steps = new int[3];

                                int[,,] ConnectionMeshGrid = new int[layer.layerSize3D[0], layer.layerSize3D[1], layer.layerSize3D[2]];
                                
                                for (int i = 0; i < layer.layerSize3D[0]; i++)
                                {
                                    //string outString = "ConnectionMeshgrid: ";
                                    for (int j = 0; j < layer.layerSize3D[1]; j++)
                                    {
                                        for (int k = 0; k < layer.layerSize3D[2]; k++)
                                        {
                                            ConnectionMeshGrid[i, j, k] = f * layer.layerSize3D[2] * layer.layerSize3D[1] * layer.layerSize3D[0] +
                                                i * layer.layerSize3D[2] * layer.layerSize3D[1] + j * layer.layerSize3D[2] + k;
                                            //outString += ConnectionMeshGrid[i, j] + " ";
                                        }
                                    }
                                    //print(outString);
                                }
                                */

                                int index = f * filterSize[0] * filterSize[1] * filterSize[2];
                                int u_pos = -padding[2];
                                int w_pos = -padding[0];
                                int h_pos = -padding[1];
                                int[] Steps = new int[3];

                                int[,,] ConnectionMeshGrid = new int[layer.layerSize3D[0], layer.layerSize3D[1], layer.layerSize3D[2]];

                                for (int k = 0; k < layer.layerSize3D[2]; k++)
                                {
                                    //string outString = "ConnectionMeshgrid: ";
                                    for (int j = 0; j < layer.layerSize3D[1]; j++)
                                    {
                                        for (int i = 0; i < layer.layerSize3D[0]; i++)
                                        {
                                            ConnectionMeshGrid[i, j, k] = f * layer.layerSize3D[2] * layer.layerSize3D[1] * layer.layerSize3D[0] +
                                                k * layer.layerSize3D[0] * layer.layerSize3D[1] + j * layer.layerSize3D[0] + i;
                                            //outString += ConnectionMeshGrid[i, j] + " ";
                                        }
                                    }
                                    //print(outString);
                                }

                                Steps = previousLayer.layerSize3D;
                                /*
                                for (int d = 0; d < Steps[0]; d++) // previous node (y)
                                {
                                    for (int g = 0; g < Steps[1]; g++) // previous node (x) (each incremenet applies a stride to pos)
                                    {
                                        for (int p = 0; p < Steps[2]; p++)
                                        {
                                            //string printOutput = "connection map c=" + c + ", w=" + d + ", h=" + g + ": ";
                                            for (int h = h_pos; h < (h_pos + filterSize[0]); h++)
                                            {
                                                for (int w = w_pos; w < (w_pos + filterSize[1]); w++)
                                                {
                                                    for (int u = u_pos; u < (u_pos + filterSize[2]); u++)
                                                    {
                                                        if ((h < layer.layerSize3D[0]) & (w < layer.layerSize3D[1]) & (u < layer.layerSize3D[2]) & w >= 0 & h >= 0 & u >= 0)
                                                        {
                                                            //print("(w,h) = (" + w + "," + h + ") , f=" + f + ", c=" + c);
                                                            //print("prev node: " + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + index + ", f=" + f);
                                                            //print("connections[" + (c * (Steps[0] * Steps[1] * Steps[2]) + d * Steps[1] * Steps[2] + g * Steps[2] + p) + ", " + index + "] = " + (ConnectionMeshGrid[h, w, u]));
                                                            connections[c * (Steps[0] * Steps[1] * Steps[2]) + d * Steps[1] * Steps[2] + g * Steps[2] + p, index] = ConnectionMeshGrid[h, w, u];
                                                            connectionsInFilter[c * (Steps[0] * Steps[1] * Steps[2]) + d * Steps[1] * Steps[2] + g * Steps[2] + p, index] = c * filterSize[0] * filterSize[1] * filterSize[2] +
                                                                (h - h_pos) * filterSize[1] * filterSize[2] + (w - w_pos) * filterSize[2] + (u - u_pos);
                                                            //printOutput += "(" + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + "(f" + connectionsInFilter[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] + ") ";
                                                            index++;
                                                            total_no_connections++;
                                                        }
                                                    }
                                                }
                                            }
                                            //print(printOutput);
                                            u_pos += stride[2];
                                            index = f * filterSize[0] * filterSize[1] * filterSize[2];
                                        }
                                        w_pos += stride[1];
                                        u_pos = -padding[2];
                                    }
                                    h_pos += stride[0];
                                    w_pos = -padding[1];
                                }
                                */



                                for (int p = 0; p < Steps[2]; p++)
                                {
                                    for (int g = 0; g < Steps[1]; g++) // previous node (x) (each incremenet applies a stride to pos)
                                    {
                                        for (int d = 0; d < Steps[0]; d++) // previous node (y)
                                        {
                                            for (int u = u_pos; u < (u_pos + filterSize[2]); u++)
                                            {
                                                //string printOutput = "connection map c=" + c + ", w=" + d + ", h=" + g + ": ";
                                                for (int h = h_pos; h < (h_pos + filterSize[1]); h++)
                                                {
                                                    for (int w = w_pos; w < (w_pos + filterSize[0]); w++)
                                                    {
                                                        if ((h < layer.layerSize3D[1]) & (w < layer.layerSize3D[0]) & (u < layer.layerSize3D[2]) & w >= 0 & h >= 0 & u >= 0)
                                                        {
                                                            //print("(w,h) = (" + w + "," + h + ") , f=" + f + ", c=" + c);
                                                            //print("prev node: " + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ", index: " + index + ", f=" + f);
                                                            //print("connections[" + (c * (Steps[0] * Steps[1] * Steps[2]) + d * Steps[1] * Steps[2] + g * Steps[2] + p) + ", " + index + "] = " + (ConnectionMeshGrid[h, w, u]));
                                                            connections[c * (Steps[0] * Steps[1] * Steps[2]) + p * Steps[1] * Steps[0] + g * Steps[0] + d, index] = ConnectionMeshGrid[w, h, u];
                                                            connectionsInFilter[c * (Steps[0] * Steps[1] * Steps[2]) + p * Steps[1] * Steps[0] + g * Steps[0] + d, index] = c * filterSize[0] * filterSize[1] * filterSize[2] +
                                                                (u - u_pos) * filterSize[1] * filterSize[0] + (h - h_pos) * filterSize[0] + (w - w_pos);
                                                            //printOutput += "(" + (c * (Steps[0] * Steps[1]) + d * Steps[1] + g) + ")" + connections[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index].ToString() + "(f" + connectionsInFilter[c * (Steps[0] * Steps[1]) + d * Steps[1] + g, index] + ") ";
                                                            index++;
                                                            total_no_connections++;
                                                        }
                                                    }
                                                }
                                            }
                                            //print(printOutput);
                                            w_pos += stride[0];
                                            index = f * filterSize[0] * filterSize[1] * filterSize[2];
                                        }
                                        h_pos += stride[1];
                                        w_pos = -padding[0];
                                    }
                                    u_pos += stride[2];
                                    h_pos = -padding[1];
                                }
                            }
                            for (int i = 0; i < previousLayer.layerSize3D[0] * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2]; i++)
                            {
                                string conns_string = "connections[" + i + ", :] = [";
                                for (int j = 0; j < no_filters * filterSize[0] * filterSize[1] * filterSize[2]; j++)
                                {
                                    conns_string += connections[i, j] + ", ";
                                    //connectionsInFilter[i, j] = -1;
                                }
                                //print(conns_string + "]");
                            }
                            int[,] connections_rev;
                            int[,] connectionsInFilter_rev;

                            int N_connections_per_node_j = Mathf.CeilToInt((float)filterSize[0] / (float)stride[0]) *
                                Mathf.CeilToInt((float)filterSize[1] / (float)stride[1]) *
                                Mathf.CeilToInt((float)filterSize[2] / (float)stride[2]);
                            connections_rev = new int[layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2], N_connections_per_node_j];
                            connectionsInFilter_rev = new int[layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2], N_connections_per_node_j];
                            for (int j=0; j < layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]; j++)
                                for (int conn=0; conn < N_connections_per_node_j; conn++)
                                {
                                    connections_rev[j, conn] = -1;
                                    connectionsInFilter_rev[j, conn] = -1;
                                }
                            //print("Conns per node: " + N_connections_per_node + ", (rev) " + N_connections_per_node_j);
                            int[] connN_j = new int[layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]];
                            for (int i=0; i < previousLayer.layerSize3D[0]* previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2]; i++)
                            {
                                //print("--- source node i=" + i);
                                for (int conn=0; conn < N_connections_per_node; conn++)
                                {
                                    int connected_node_j = connections[i, conn];
                                    int cif_j = connectionsInFilter[i, conn];
                                    if ((connected_node_j > -1) & (connected_node_j < (layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2])))
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

                            #region TConv3D_connectionMap_debugging
                            /*
                            print("-------- Normal:");
                            print("Connections per node: " + N_connections_per_node);
                            print("Connections: ");
                            for (int i = 0; i < previousLayer.layerSize3D[0] * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2]; i++)
                            {
                                string conString = "conn [" + i + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node; conn++)
                                {
                                    conString += connections[i, conn] + ", ";
                                }
                                print(conString);
                            }
                            print("CIF: ");
                            for (int i = 0; i < previousLayer.layerSize3D[0] * previousLayer.layerSize3D[1] * previousLayer.layerSize3D[2]; i++)
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
                            for (int j = 0; j < layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]; j++)
                            {
                                string conString = "conn [" + j + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node_j; conn++)
                                {
                                    conString += connections_rev[j, conn] + ", ";
                                }
                                print(conString);
                            }
                            print("CIF: ");
                            for (int j = 0; j < layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2]; j++)
                            {
                                string cifString = "cif [" + j + ",:] = ";
                                for (int conn = 0; conn < N_connections_per_node_j; conn++)
                                {
                                    cifString += connectionsInFilter_rev[j, conn] + ", ";
                                }
                                print(cifString);
                            }
                            */
                            #endregion

                            connections = connections_rev;
                            connectionsInFilter = connectionsInFilter_rev;
                            N_connections_per_node = N_connections_per_node_j;

                        }

                        break;
                    }
            }

        }


        public void BuildTrackers3D(Noedify.Layer layer)
        {
            filterTrack = new int[layer.layerSize];
            channelTrack = new int[layer.layerSize];
            nodeTrack = new int[layer.layerSize];

            int nodesPerFilter = layer.layerSize3D[0] * layer.layerSize3D[1] * layer.layerSize3D[2];

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
