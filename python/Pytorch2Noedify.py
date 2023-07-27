import os
import numpy as np
import torch
import torch.nn as nn


def ExportModel_pyTorch(model, filename):
    paramFile = open(filename, "w+")
    print("Number of layers: {}".format(len(model)))
    inputFullyConnected = 0
    is_layer_1 = 1

    for layer in model.children():
        if isinstance(layer, nn.Linear):
            print("Linear (Dense) Layer")
            [ni, nj] = np.shape((layer.weight).cpu())
            if (inputFullyConnected):
                for j in range(0, nj):
                    for i in range(0, ni):
                        paramFile.write("{},".format(layer.weights[i, j].detach().cpu().numpy()))
                    outString = outString[:-1]
                paramFile.write("*\n")
                # biases = layers.bias[1]; # get biases
                # for j in range(0,np.shape(biases)[0]):
                #      paramFile.write("{}\n".format(biases[j]))
                paramFile.write("***\n")
            paramFile.write(outString)

        elif isinstance(layer, nn.Conv2d):
            print("Convolutional 2D Layer")
            [nf, nic, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = ""
                            for j in range(0, nic):
                                outString = outString + "{},".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)
            else:
                for f in range(0, nf):
                    outString = ""
                    for c in range(0, nic):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{},".format(layer.weight[f, c, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    outString = outString + "\n"
                    paramFile.write(outString)
            paramFile.write("*\n")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            for j in range(0, n_bias):
                paramFile.write("{}\n".format(layer.bias[j]))
            paramFile.write("***\n")
            inputFullyConnected = 0
        elif isinstance(layer, nn.Conv3d):
            print("Convolutional 3D Layer")
            [nf, nic, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = ""
                            for j in range(0, nic):
                                outString = outString + "{},".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)
            else:
                for f in range(0, nf):
                    outString = ""
                    for c in range(0, nic):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{},".format(layer.weight[f, c, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    outString = outString + "\n"
                    paramFile.write(outString)
            paramFile.write("*\n")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            for j in range(0, n_bias):
                paramFile.write("{}\n".format(layer.bias[j]))
            paramFile.write("***\n")
            inputFullyConnected = 0
        elif isinstance(layer, nn.ConvTranspose2d):
            print("Convolutional Transpose 2D Layer")
            [nc, nf, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = ""
                            for j in range(0, nc):
                                outString = outString + "{},".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)
            else:
                for f in range(0, nf):
                    outString = ""
                    for c in range(0, nc):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{},".format(layer.weight[c, f, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    outString = outString + "\n"
                    paramFile.write(outString)
            paramFile.write("*\n")
            # BIASES
            if (np.shape(layer.bias)):
                n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
                outString = ""
                for j in range(0, n_bias):
                    outString = outString + "{},".format(layer.bias[j])
                outString = outString[:-1]
                outString = outString + "\n"
                paramFile.write(outString)
            paramFile.write("***\n")
            inputFullyConnected = 0
        elif isinstance(layer, nn.BatchNorm2d):
            print("Batch Normalization 2D Layer")
            nj = np.shape(layer.weight.detach().cpu().numpy())
            nj = np.squeeze(nj)
            outString = ""
            for j in range(0, nj):
                outString = outString + "{},".format(layer.weight[j].detach().cpu().numpy())
                # paramFile.write("{},".format(layer.weights[i,j]))
            outString = outString[:-1]
            outString = outString + "\n";
            paramFile.write(outString)
            paramFile.write("*\n")
            outString = ""
            for j in range(0, nj):
                outString = outString + "{},".format(layer.bias[j].detach().cpu().numpy())
            outString = outString[:-1]
            outString = outString + "\n";
            paramFile.write(outString)
            paramFile.write("***\n")
        else:
            print("No parameters exported for layer {}".format(layer))
        is_layer_1 = 0

    paramFile.close()
    print("Complete.")


def ExportCompilableModel_pyTorch(model, filename, modelCode=None):

    no_export_layer = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.Conv2d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.Conv3d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.ConvTranspose2d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.ConvTranspose3d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.BatchNorm2d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.BatchNorm3d):
            no_export_layer = no_export_layer + 1
        else:
            print("No parameters exported for layer {}".format(layer))
        is_layer_1 = 0
    print("Exporting {} layers".format(no_export_layer))

    paramFile = open(filename, "w+")
    print("Number of layers: {}".format(len(model)))
    inputFullyConnected = 0
    is_layer_1 = 1
    lyr = 1
    paramFile.write(
        "\tpublic static NetParam[] netParams = new NetParam[{}] {{new NetParam(),\n\t\t".format(no_export_layer + 1))
    for layer in model.children():
        lyr = lyr + 1
        ExportCompilableLayer_pyTorch(layer, lyr, paramFile)
        is_layer_1 = 0
    paramFile.write("\n\t};")
    if modelCode is not None:
        paramFile.write("\n\tpublic static string model_code = \"" + modelCode + "\";\n")
    paramFile.close()
    print("Complete.")


def ExportCompilableModel_SingleLayer_pyTorch(layer, filename):

    paramFile = open(filename, "w+")
    lyr = 1
    paramFile.write(
        "\tpublic static NetParam[] netParams = new NetParam[{}] {{new NetParam(),\n\t\t".format(2))

    ExportCompilableLayer_pyTorch(layer, lyr, paramFile)
    paramFile.write("\n\t};")
    paramFile.close()
    print("Complete.")


def ExportCompilableLayer_pyTorch(layer, lyr, output_file):
    inputFullyConnected = 0

    if isinstance(layer, nn.Linear):
        output_file.write("new NetParam(")
        print("Linear (Dense) Layer")
        [ni, nj] = np.shape((layer.weight).cpu())
        if (inputFullyConnected):
            for j in range(0, nj):
                for i in range(0, ni):
                    output_file.write("{},".format(layer.weights[i, j].detach().cpu().numpy()))
                outString = outString[:-1]
            output_file.write("*\n")
            output_file.write("***\n")
        output_file.write(outString)

    elif isinstance(layer, nn.Conv2d):
        output_file.write("new NetParam(")
        print("Convolutional 2D Layer")
        [nf, nic, ny, nx] = np.shape((layer.weight).cpu())
        if inputFullyConnected == 1:
            for f in range(0, nf):
                for iy in range(0, ny):
                    for ix in range(0, nx):
                        outString = ""
                        for j in range(0, nic):
                            outString = outString + "{}f,".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                        outString = outString[:-1]
                        outString = outString + "\n"
                        output_file.write(outString)

        else:
            output_file.write(
                "new float[{},{}] {{ // ({}) Convolutional 2D - Weights\n\t\t\t".format(nf, nic * ny * nx, lyr))
            for f in range(0, nf):
                outString = "{"
                for c in range(0, nic):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = outString + "{}f,".format(layer.weight[f, c, iy, ix].detach().cpu().numpy())
                outString = outString[:-1]
                output_file.write(outString + "},\n\t\t\t")
            output_file.write("},\n\t\t")
        # BIASES
        n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
        output_file.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
        outString = ""
        for j in range(0, n_bias):
            outString = outString + "{}f,".format(layer.bias[j])
        outString = outString[:-1]
        output_file.write(outString)
        output_file.write("}),\n\t\t")
        inputFullyConnected = 0
    elif isinstance(layer, nn.Conv3d):
        output_file.write("new NetParam(")
        print("Convolutional 3D Layer")

        [nf, nic, nx, ny, nz] = np.shape((layer.weight).cpu())
        print(np.shape((layer.weight).cpu()))
        if inputFullyConnected == 1:

            [nf, nic, nz, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iz in range(0,nz):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = ""
                                for j in range(0, nic):
                                    #### WARNING: index dimension order is suspect!!
                                    outString = outString + "{}f,".format(layer.weight[j, f, iz, iy, ix].detach().cpu().numpy())
                                outString = outString[:-1]
                                outString = outString + "\n"
                                output_file.write(outString)

        else:
            output_file.write(
                "new float[{},{}] {{ // ({}) Convolutional 3D - Weights\n\t\t\t".format(nf, nic * nz * ny * nx, lyr))
            for f in range(0, nf):
                outString = "{"
                for c in range(0, nic):
                    for iz in range(0, nz):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{}f,".format(layer.weight[f, c, ix, iy, iz].detach().cpu().numpy())
                outString = outString[:-1]
                output_file.write(outString + "},\n\t\t\t")
            output_file.write("},\n\t\t")
        # BIASES
        n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
        output_file.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
        outString = ""
        for j in range(0, n_bias):
            outString = outString + "{}f,".format(layer.bias[j])
        outString = outString[:-1]
        output_file.write(outString)
        output_file.write("}),\n\t\t")
        inputFullyConnected = 0
    elif isinstance(layer, nn.ConvTranspose2d):
        output_file.write("new NetParam(")
        print("Convolutional Transpose 2D Layer")
        [nc, nf, ny, nx] = np.shape((layer.weight).cpu())

        if inputFullyConnected == 1:
            for f in range(0, nf):
                for iy in range(0, ny):
                    for ix in range(0, nx):
                        outString = ""
                        for j in range(0, nc):
                            outString = outString + "{}f,".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                        outString = outString[:-1]
                        outString = outString + "\n"
                        output_file.write(outString)
        else:
            output_file.write(
                "new float[{},{}] {{ // ({}) Transpose Convolutional 2D - Weights\n\t\t\t".format(nf, nc * ny * nx,
                                                                                                  lyr))
            for f in range(0, nf):
                outString = "{"
                for c in range(0, nc):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = outString + "{}f,".format(layer.weight[c, f, iy, ix].detach().cpu().numpy())
                outString = outString[:-1]
                output_file.write(outString + "},\n\t\t\t")
            output_file.write("},\n\t\t")
        # BIASES
        n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
        output_file.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
        outString = ""
        for j in range(0, n_bias):
            outString = outString + "{}f,".format(layer.bias[j])
        outString = outString[:-1]
        output_file.write(outString)
        output_file.write("}),\n\t\t")
        inputFullyConnected = 0
    elif isinstance(layer, nn.ConvTranspose3d):
        output_file.write("new NetParam(")
        print("Convolutional Transpose 3D Layer")
        [nc, nf, nx, ny, nz] = np.shape((layer.weight).cpu())

        if inputFullyConnected == 1:
            for f in range(0, nf):
                for iz in range(0, nz):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = ""
                            for j in range(0, nc):
                                #### WARNING: index dimension order is suspect!!
                                outString = outString + "{}f,".format(layer.weight[j, f, iz, iy, ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            output_file.write(outString)
        else:
            output_file.write(
                "new float[{},{}] {{ // ({}) Transpose Convolutional 3D - Weights\n\t\t\t".format(nf,
                                                                                                  nc * nz * ny * nx,
                                                                                                  lyr))
            for f in range(0, nf):
                outString = "{"
                for c in range(0, nc):
                    for iz in range(0, nz):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{}f,".format(layer.weight[c, f, ix, iy, iz].detach().cpu().numpy())
                outString = outString[:-1]
                output_file.write(outString + "},\n\t\t\t")
            output_file.write("},\n\t\t")
        # BIASES
        n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
        output_file.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
        outString = ""
        for j in range(0, n_bias):
            outString = outString + "{}f,".format(layer.bias[j])
        outString = outString[:-1]
        output_file.write(outString)
        output_file.write("}),\n\t\t")
        inputFullyConnected = 0
    elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
        output_file.write("new NetParam(")
        print("Batch Normalization Layer")
        nj = np.shape(layer.weight.detach().cpu().numpy())
        nj = np.squeeze(nj)
        output_file.write("new float[1,{}] {{ // ({}) Batch Normalization - Weights\n\t\t\t".format(nj, lyr))
        outString = "{"
        for j in range(0, nj):
            outString = outString + "{}f,".format(layer.weight[j].detach().cpu().numpy())
            # output_file.write("{},".format(layer.weights[i,j]))
        outString = outString[:-1]
        outString = outString + "}},\n\t\t\t";
        output_file.write(outString)
        # BIASES
        n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
        output_file.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
        outString = ""
        for j in range(0, n_bias):
            outString = outString + "{}f,".format(layer.bias[j].detach().cpu().numpy())
        outString = outString[:-1]
        output_file.write(outString + "}),\n\t\t")
        inputFullyConnected = 0
    else:
        print("No parameters exported for layer {}".format(layer))


def GetModelCode(model, input_shape):

    code_str = ""
    if len(input_shape)==1:
        code_str += "1_{}_".format(input_shape)
    elif (len(input_shape)==3):
        code_str += "2_{},{},{}_".format(input_shape[1], input_shape[2], input_shape[0])
    elif (len(input_shape)==4):
        code_str += "3_{},{},{},{}_".format(input_shape[1], input_shape[2], input_shape[3], input_shape[0])
    else:
        print("(GetModelCode) invalid input_shape: {}".format(input_shape))
        return
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            code_str += "a"
        elif isinstance(layer, nn.Conv2d):
            code_str += "b"
        elif isinstance(layer, nn.BatchNorm2d):
            code_str += "c"
        elif isinstance(layer, nn.ConvTranspose2d):
            code_str += "d"
        elif isinstance(layer, nn.Conv3d):
            code_str += "e"
        elif isinstance(layer, nn.BatchNorm3d):
            code_str += "f"
        elif isinstance(layer, nn.ConvTranspose3d):
            code_str += "g"
    code_str += "_"
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            code_str += ""
        elif isinstance(layer, nn.Conv2d):
            code_str += "{},{},{},{},{},{},{}_".format(layer.kernel_size[0], layer.kernel_size[1], layer.out_channels,
                                                       layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
        elif isinstance(layer, nn.BatchNorm2d):
            code_str += "{}_".format(layer.eps)
        elif isinstance(layer, nn.ConvTranspose2d):
            code_str += "{},{},{},{},{},{},{}_".format(layer.kernel_size[0], layer.kernel_size[1], layer.out_channels,
                                                       layer.stride[0], layer.stride[1], layer.padding[0],
                                                       layer.padding[1])
        elif isinstance(layer, nn.Conv3d):
            code_str += "{},{},{},{},{},{},{},{},{},{}_".format(layer.kernel_size[0], layer.kernel_size[1], layer.kernel_size[2],
                                                                layer.out_channels, layer.stride[0], layer.stride[1],
                                                                layer.stride[2], layer.padding[0], layer.padding[1],
                                                                layer.padding[2])
        elif isinstance(layer, nn.BatchNorm3d):
            code_str += "{}_".format(layer.eps)
        elif isinstance(layer, nn.ConvTranspose3d):
            code_str += "{},{},{},{},{},{},{},{},{},{}_".format(layer.kernel_size[0], layer.kernel_size[1], layer.kernel_size[2],
                                                                layer.out_channels, layer.stride[0], layer.stride[1],
                                                                layer.stride[2], layer.padding[0], layer.padding[1],
                                                                layer.padding[2])
    prev_layer_hidden = False
    for layer in model.children():
        if isinstance(layer, nn.ReLU):
            code_str += "R"
            prev_layer_hidden = False
        elif isinstance(layer, nn.Sigmoid):
            code_str += "S"
            prev_layer_hidden = False
        elif isinstance(layer, nn.Tanh):
            code_str += "T"
            prev_layer_hidden = False
        elif prev_layer_hidden:
            code_str += "L"
            prev_layer_hidden = True
        else:
            prev_layer_hidden = True
    if prev_layer_hidden:
        code_str += "L"


    return code_str



'''
## original all-in-one compilable parameter export (before breaking into ExportCompilableLayer_pyTorch)
def ExportCompilableModel_pyTorch(model, filename):

    no_export_layer = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.Conv2d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.Conv3d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.ConvTranspose2d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.ConvTranspose3d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.BatchNorm2d):
            no_export_layer = no_export_layer + 1
        elif isinstance(layer, nn.BatchNorm3d):
            no_export_layer = no_export_layer + 1
        else:
            print("No parameters exported for layer {}".format(layer))
        is_layer_1 = 0
    print("Exporting {} layers".format(no_export_layer))

    paramFile = open(filename, "w+")
    print("Number of layers: {}".format(len(model)))
    inputFullyConnected = 0
    is_layer_1 = 1
    lyr = 1
    paramFile.write(
        "\tpublic static NetParam[] netParams = new NetParam[{}] {{new NetParam(),\n\t\t".format(no_export_layer + 1))
    for layer in model.children():
        lyr = lyr + 1
        if isinstance(layer, nn.Linear):
            paramFile.write("new NetParam(")
            print("Linear (Dense) Layer")
            [ni, nj] = np.shape((layer.weight).cpu())
            if (inputFullyConnected):
                for j in range(0, nj):
                    for i in range(0, ni):
                        paramFile.write("{},".format(layer.weights[i, j].detach().cpu().numpy()))
                    outString = outString[:-1]
                paramFile.write("*\n")
                paramFile.write("***\n")
            paramFile.write(outString)

        elif isinstance(layer, nn.Conv2d):
            paramFile.write("new NetParam(")
            print("Convolutional 2D Layer")
            [nf, nic, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = ""
                            for j in range(0, nic):
                                outString = outString + "{}f,".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)

            else:
                paramFile.write(
                    "new float[{},{}] {{ // ({}) Convolutional 2D - Weights\n\t\t\t".format(nf, nic * ny * nx, lyr))
                for f in range(0, nf):
                    outString = "{"
                    for c in range(0, nic):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{}f,".format(layer.weight[f, c, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    paramFile.write(outString + "},\n\t\t\t")
                paramFile.write("},\n\t\t")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            paramFile.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
            outString = ""
            for j in range(0, n_bias):
                outString = outString + "{}f,".format(layer.bias[j])
            outString = outString[:-1]
            paramFile.write(outString)
            paramFile.write("}),\n\t\t")
            inputFullyConnected = 0
        elif isinstance(layer, nn.Conv3d):
            paramFile.write("new NetParam(")
            print("Convolutional 3D Layer")
            [nf, nic, nz, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iz in range(0,nz):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = ""
                                for j in range(0, nic):
                                    outString = outString + "{}f,".format(layer.weight[j, f, iz, iy, ix].detach().cpu().numpy())
                                outString = outString[:-1]
                                outString = outString + "\n"
                                paramFile.write(outString)

            else:
                paramFile.write(
                    "new float[{},{}] {{ // ({}) Convolutional 3D - Weights\n\t\t\t".format(nf, nic * nz * ny * nx, lyr))
                for f in range(0, nf):
                    outString = "{"
                    for c in range(0, nic):
                        for iz in range(0, nz):
                            for iy in range(0, ny):
                                for ix in range(0, nx):
                                    outString = outString + "{}f,".format(layer.weight[f, c, iz, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    paramFile.write(outString + "},\n\t\t\t")
                paramFile.write("},\n\t\t")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            paramFile.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
            outString = ""
            for j in range(0, n_bias):
                outString = outString + "{}f,".format(layer.bias[j])
            outString = outString[:-1]
            paramFile.write(outString)
            paramFile.write("}),\n\t\t")
            inputFullyConnected = 0
        elif isinstance(layer, nn.ConvTranspose2d):
            paramFile.write("new NetParam(")
            print("Convolutional Transpose 2D Layer")
            [nc, nf, ny, nx] = np.shape((layer.weight).cpu())

            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iy in range(0, ny):
                        for ix in range(0, nx):
                            outString = ""
                            for j in range(0, nc):
                                outString = outString + "{}f,".format(layer.weight[j, f, iy, ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)
            else:
                paramFile.write(
                    "new float[{},{}] {{ // ({}) Transpose Convolutional 2D - Weights\n\t\t\t".format(nf, nc * ny * nx,
                                                                                                      lyr))
                for f in range(0, nf):
                    outString = "{"
                    for c in range(0, nc):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = outString + "{}f,".format(layer.weight[c, f, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    paramFile.write(outString + "},\n\t\t\t")
                paramFile.write("},\n\t\t")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            paramFile.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
            outString = ""
            for j in range(0, n_bias):
                outString = outString + "{}f,".format(layer.bias[j])
            outString = outString[:-1]
            paramFile.write(outString)
            paramFile.write("}),\n\t\t")
            inputFullyConnected = 0
        elif isinstance(layer, nn.ConvTranspose3d):
            paramFile.write("new NetParam(")
            print("Convolutional Transpose 3D Layer")
            [nc, nf, nz, ny, nx] = np.shape((layer.weight).cpu())

            if inputFullyConnected == 1:
                for f in range(0, nf):
                    for iz in range(0, nz):
                        for iy in range(0, ny):
                            for ix in range(0, nx):
                                outString = ""
                                for j in range(0, nc):
                                    outString = outString + "{}f,".format(layer.weight[j, f, iz, iy, ix].detach().cpu().numpy())
                                outString = outString[:-1]
                                outString = outString + "\n"
                                paramFile.write(outString)
            else:
                paramFile.write(
                    "new float[{},{}] {{ // ({}) Transpose Convolutional 3D - Weights\n\t\t\t".format(nf,
                                                                                                      nc * nz * ny * nx,
                                                                                                      lyr))
                for f in range(0, nf):
                    outString = "{"
                    for c in range(0, nc):
                        for iz in range(0, nz):
                            for iy in range(0, ny):
                                for ix in range(0, nx):
                                    outString = outString + "{}f,".format(layer.weight[c, f, iz, iy, ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    paramFile.write(outString + "},\n\t\t\t")
                paramFile.write("},\n\t\t")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            paramFile.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
            outString = ""
            for j in range(0, n_bias):
                outString = outString + "{}f,".format(layer.bias[j])
            outString = outString[:-1]
            paramFile.write(outString)
            paramFile.write("}),\n\t\t")
            inputFullyConnected = 0
        elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
            paramFile.write("new NetParam(")
            print("Batch Normalization Layer")
            nj = np.shape(layer.weight.detach().cpu().numpy())
            nj = np.squeeze(nj)
            paramFile.write("new float[1,{}] {{ // ({}) Batch Normalization - Weights\n\t\t\t".format(nj, lyr))
            outString = "{"
            for j in range(0, nj):
                outString = outString + "{}f,".format(layer.weight[j].detach().cpu().numpy())
                # paramFile.write("{},".format(layer.weights[i,j]))
            outString = outString[:-1]
            outString = outString + "}},\n\t\t\t";
            paramFile.write(outString)
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            paramFile.write("new float[{}] {{ // Biases\n\t\t\t".format(n_bias))
            outString = ""
            for j in range(0, n_bias):
                outString = outString + "{}f,".format(layer.bias[j].detach().cpu().numpy())
            outString = outString[:-1]
            paramFile.write(outString + "}),\n\t\t")
            inputFullyConnected = 0
        else:
            print("No parameters exported for layer {}".format(layer))
        is_layer_1 = 0
    paramFile.write("\n\t};")
    paramFile.close()
    print("Complete.")
'''