def ExportModel_pyTorch(model, filename):
    import os
    import numpy as np

    paramFile = open(filename,"w+")
    print("Number of layers: {}".format(len(model)))
    inputFullyConnected = 0
    is_layer_1 = 1

    for layer in model.children():
        if isinstance(layer, nn.Linear):
            print("Linear (Dense) Layer")
            [ni, nj] = np.shape((layer.weight).cpu())
            if (inputFullyConnected):
                for j in range(0,nj):
                    for i in range(0,ni):
                        paramFile.write("{},".format(layer.weights[i,j].detach().cpu().numpy()))
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
                for f in range(0,nf):
                    for iy in range(0,ny):
                        for ix in range(0,nx):
                            outString = ""
                            for j in range(0,nic):
                                outString = outString + "{},".format(layer.weight[j,f,iy,ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)
            else:    
              for f in range(0,nf):
                  outString = ""
                  for c in range(0,nic):
                      for iy in range(0,ny):
                          for ix in range(0,nx):
                              outString = outString + "{},".format(layer.weight[f,c,iy,ix].detach().cpu().numpy())
                  outString = outString[:-1]
                  outString = outString + "\n"
                  paramFile.write(outString)
            paramFile.write("*\n")
            # BIASES
            n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
            for j in range(0,n_bias):
              paramFile.write("{}\n".format(layer.bias[j]))
            paramFile.write("***\n")
            inputFullyConnected = 0
        elif isinstance(layer,nn.ConvTranspose2d):
            print("Convolutional Transpose 2D Layer")
            [nc, nf, ny, nx] = np.shape((layer.weight).cpu())
            if inputFullyConnected == 1:
                for f in range(0,nf):
                    for iy in range(0,ny):
                        for ix in range(0,nx):
                            outString = ""
                            for j in range(0,nc):
                                outString = outString + "{},".format(layer.weight[j,f,iy,ix].detach().cpu().numpy())
                            outString = outString[:-1]
                            outString = outString + "\n"
                            paramFile.write(outString)
            else:    
                for f in range(0,nf):
                    outString = ""
                    for c in range(0,nc):
                        for iy in range(0,ny):
                            for ix in range(0,nx):
                                outString = outString + "{},".format(layer.weight[c,f,iy,ix].detach().cpu().numpy())
                    outString = outString[:-1]
                    outString = outString + "\n"
                    paramFile.write(outString)
            paramFile.write("*\n")
            # BIASES
            if np.shape(layer.bias):
              n_bias = np.squeeze(np.shape(layer.bias.detach().cpu().numpy()))
              outString = ""
              for j in range(0,n_bias):
                outString = outString + "{},".format(layer.bias[j])
              outString = outString[:-1]
              outString = outString + "\n"
              paramFile.write(outString)
              
            paramFile.write("***\n")
            inputFullyConnected = 0
        elif isinstance(layer,nn.BatchNorm2d):
            print("Batch Normalization 2D Layer")
            nj = np.shape(layer.weight.detach().cpu().numpy())
            nj = np.squeeze(nj)
            outString = ""
            for j in range(0,nj):
                outString = outString + "{},".format(layer.weight[j].detach().cpu().numpy())
                #paramFile.write("{},".format(layer.weights[i,j]))
            outString = outString[:-1]
            outString = outString + "\n";
            paramFile.write(outString)
            paramFile.write("*\n")
            outString = ""
            for j in range(0,nj):
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