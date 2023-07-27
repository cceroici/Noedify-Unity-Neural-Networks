#define NOEDIFY_NORELEASE

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Noedify_Utils
{
    public static int ConvertOneHotToInt(float[] oneHot)
    {
        int maxInt = -1;
        float maxVal = -10;
        for (int i = 0; i < oneHot.Length; i++)
        {
            if (oneHot[i] > maxVal)
            {
                maxVal = oneHot[i];
                maxInt = i;
            }
        }
        return maxInt;
    }


    public static int ArgMax(float[] oneHot)
    {
        /// Performs argmax operation on float vector, returning integer index
        int maxInt = -1;
        float maxVal = -10;
        for (int i = 0; i < oneHot.Length; i++)
        {
            if (oneHot[i] > maxVal)
            {
                maxVal = oneHot[i];
                maxInt = i;
            }
        }
        return maxInt;
    }


    public static float[] Ones(int shape){
        // Generate an array of ones
        float[] oneArray = new float[shape];
        for (int i=0; i < shape; i++)
            oneArray[i] = 1.0f;
        return oneArray;
    }

    public static float[,] Ones(int shape_x, int shape_y){
        // Generate an array of ones
        float[,] oneArray = new float[shape_x, shape_y];
        for (int i=0; i < shape_x; i++)
            for (int j=0; j<shape_y; j++)
                oneArray[i,j] = 1.0f;
        return oneArray;
    }

    public static float[,,] Ones(int shape_x, int shape_y, int shape_z){
        // Generate an array of ones
        float[,,] oneArray = new float[shape_x, shape_y, shape_z];
        for (int i=0; i < shape_x; i++)
            for (int j=0; j<shape_y; j++)
                for (int k=0; j<shape_z; k++)
                    oneArray[i,j,k] = 1.0f;
        return oneArray;
    }

    public static float[] FlattenDataset(float[,] matrix)
    {
        int c = matrix.GetLength(0);
        int w = matrix.GetLength(1);
        float[] flattenedMatrix = new float[c * w];
        for (int i = 0; i < c; i++)
            for (int j = 0; j < w; j++)
                flattenedMatrix[i * w + j] = matrix[i, j];
        return flattenedMatrix;
    }
    public static float[] FlattenDataset(float[,,] matrix, int no_channels)
    {
        int h = matrix.GetLength(1);
        int w = matrix.GetLength(2); 
        float[] flattenedMatrix = new float[no_channels * h * w];
        for (int i = 0; i < no_channels; i++)
            for (int j = 0; j < h; j++)
                for (int k = 0; k < w; k++)
                    flattenedMatrix[i * h * w + j * w + k] = matrix[i, j, k];
        return flattenedMatrix;
    }
    public static float[] FlattenDataset(float[,,] matrix)
    {
        int w = matrix.GetLength(0);
        int h = matrix.GetLength(1); 
        int d = matrix.GetLength(2);

        float[] flattenedMatrix = new float[d * h * w];
        for (int iz = 0; iz < d; iz++)
            for (int iy = 0; iy < h; iy++)
                for (int ix = 0; ix < w; ix++)
                    flattenedMatrix[iz * h * w + iy * w + ix] = matrix[ix, iy, iz];
        return flattenedMatrix;
    }
    public static float[] FlattenDataset(float[,,,] matrix)
    {
        int c = matrix.GetLength(0);
        int w = matrix.GetLength(1);
        int h = matrix.GetLength(2);
        int d = matrix.GetLength(3);
        float[] flattenedMatrix = new float[c * h * w * d];
        for (int ic = 0; ic < c; ic++)
            for (int iz = 0; iz < d; iz++)
                for (int iy = 0; iy < h; iy++)
                    for (int ix=0; ix < w; ix++)
                        flattenedMatrix[ic * w * h * d + iz * w * h + iy * w + ix ] = matrix[ic, ix, iy, iz];
        return flattenedMatrix;
    }
    public static List<float[]> FlattenDataset(List<float[,,]> matrix)
    {
        int c = matrix[0].GetLength(0);
        int h = matrix[0].GetLength(1);
        int w = matrix[0].GetLength(2);
        List<float[]> flattenedMatrixList = new List<float[]>();
        for (int n = 0; n < matrix.Count; n++) {
            float[] flattenedMatrix = new float[c * h * w];
            for (int i = 0; i < c; i++)
                for (int j = 0; j < h; j++)
                    for (int k = 0; k < w; k++)
                        flattenedMatrix[i * h * w + j * w + k] = matrix[n][i, j, k];
            flattenedMatrixList.Add(flattenedMatrix);
        }
        return flattenedMatrixList;
    }
    public static List<float[]> FlattenDataset(List<Noedify.InputArray> matrix)
    {
        
        List<float[]> flattenedMatrixList = new List<float[]>();
        for (int n = 0; n < matrix.Count; n++)
            flattenedMatrixList.Add(matrix[n].FlattenArray());
        return flattenedMatrixList;
    }

    public static float[,,,] UnflattenArray(float[] array, int c, int w, int h, int d, int p)
    {
        float[,,,] reshaped = new float[c, w, h, d];
        for (int ic = 0; ic < c; ic++)
            for (int id = 0; id < d; id++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                        reshaped[ic, iw, ih, id] = array[ic * w * h * d + id * h * w + ih * w + iw];


        return reshaped;
                        }
    public static float[,,] UnflattenArray(float[] array, int w, int h, int d)
    {
        float[,,] reshaped = new float[w, h, d];
        for (int id = 0; id < d; id++)
            for (int ih = 0; ih < h; ih++)
                for (int iw = 0; iw < w; iw++)
                    reshaped[iw, ih, id] = array[id * h * w + ih * w + iw];


        return reshaped;
    }

    public static int[] Shuffle(int[] inputArray, int seed = -1)
    {
        if (inputArray.Length == 1)
            return inputArray;
        else
        {
            if (seed >= 0)
            {
                Random.InitState(seed);
            }
            int temp;
            for (int i = 0; i < inputArray.Length; i++)
            {
                int rnd = Random.Range(0, inputArray.Length);
                temp = inputArray[rnd];
                inputArray[rnd] = inputArray[i];
                inputArray[i] = temp;
            }
            return inputArray;
        }
    }

    public static float[,] RotateImage90(float[,] imageData)
    {
        // Rotate 2D array clockwise/counter-clockwise
        int h = imageData.GetLength(0);
        int w = imageData.GetLength(1);

        float[,] imageData_rot = new float[h, w];


        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++)
            {
                imageData_rot[i, j] = imageData[j, i];
            }
        return imageData_rot;
    }

    public static float[,] FlipImage(float[,] imageData, bool horizontal = true)
    {
        // Flip 2D array horizontall/vertically
        int h = imageData.GetLength(0);
        int w = imageData.GetLength(1);
        float[,] imageData_flip = new float[h, w];
        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++)
            {
                if (horizontal)
                    imageData_flip[i, j] = imageData[w - i - 1, j];
                else
                    imageData_flip[i, j] = imageData[w, h - j - 1];
            }
        return imageData_flip;
    }

    public static float[,] FlipLR(float[,] array)
    {
        // Flip 2D array horizontally
        int h = array.GetLength(0);
        int w = array.GetLength(1);
        float[,] array_flip = new float[h, w];
        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++)
            {
                array_flip[i, j] = array[w - i - 1, j];
            }
        return array_flip;
    }

    public static float[,] FlipUD(float[,] array)
    {
        // Flip 2D array vertically
        int h = array.GetLength(0);
        int w = array.GetLength(1);
        float[,] array_flip = new float[h, w];
        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++)
            {
                array_flip[i, j] = array[j, h - j - 1];
            }
        return array_flip;
    }

    public static float[,,,] AddSingularDim(float [,,] array)
    {
        // Add single dimention to 1D or 2D array
        int h = array.GetLength(0);
        int w = array.GetLength(1);
        int d = array.GetLength(2);
        float[,,,] newArray = new float[1, h, w, d];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int k=0; k < d; k++)
                    newArray[0, i, j, k] = array[i, j, k];
        return newArray;
    }
    public static float[,,] AddSingularDim(float[,] array)
    {
        // Add single dimention to 1D or 2D array
        int h = array.GetLength(0);
        int w = array.GetLength(1);
        float[,,] newArray = new float[1, h, w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                newArray[0, i, j] = array[i, j];
        return newArray;
    }
    public static float[,] AddSingularDim(float[] array)
    {
        int h = array.GetLength(0);
        float[,] newArray = new float[1, h];
        for (int i = 0; i < h; i++)
                newArray[0, i] = array[i];
        return newArray;
    }

    public static float[,,] AddTwoSingularDims(float[] array)
    {
        // Add single dimention to 1D or 2D array
        int w = array.Length;
        float[,,] newArray = new float[1, 1, w];
        for (int j = 0; j < w; j++)
            newArray[0, 0, j] = array[j];
        return newArray;
    }

    // Remove singular dimension from array
    public static float[,] SqueezeDim(float[,,] array, int squeezeIndex, int dimensionIndex = 0)
    {
        int a = array.GetLength(0);
        int b = array.GetLength(1);
        int c = array.GetLength(2);
        float[,] newArray;
        if (dimensionIndex == 0) {
            newArray = new float[b, c];
            for (int i = 0; i < b; i++)
                for (int j = 0; j < c; j++)
                    newArray[i, j] = array[squeezeIndex, i, j];
        }
        else if (dimensionIndex == 1) {
            newArray = new float[a, c];
            for (int i = 0; i < a; i++)
                for (int j = 0; j < c; j++)
                    newArray[i, j] = array[i, squeezeIndex, j];
        }
        else {
            newArray = new float[a, b];
            for (int i = 0; i < a; i++)
                for (int j = 0; j < b; j++)
                    newArray[i, j] = array[i, j, squeezeIndex];
        }
        return newArray;
    }
    public static float[] SqueezeDim(float[,] array, int squeezeIndex, int dimensionIndex = 0)
    {
        int a = array.GetLength(0);
        int b = array.GetLength(1);
        float[] newArray;
        if (dimensionIndex == 0)
        {
            newArray = new float[b];
            for (int i = 0; i < b; i++)
                    newArray[i] = array[squeezeIndex, i];
        }
        else 
        {
            newArray = new float[a];
            for (int i = 0; i < a; i++)
                    newArray[i] = array[i, squeezeIndex];
        }
        return newArray;
    }

    public static string Print2DArray(float[,] array)
    {
        string stringOutn = "";
        for (int i = 0; i < array.GetLength(0); i++)
        {
            stringOutn = "";
            for (int j = 0; j < array.GetLength(1); j++)
            {
                stringOutn += array[i, j] + ",";
            }
        }
        return stringOutn;
    }

    public static void PrintArray<T>(T[] array, string variableName)
    {
        string stringOutn = variableName + ": ";
        for (int i = 0; i < array.Length; i++)
        {
            stringOutn += array[i] + ",";
        }
        Debug.Log(stringOutn);
    }

    public static void WriteImage(string fileName, Color[,] imgData, int[] dims)
    {

        Texture2D sampleTexture = new Texture2D(dims[0], dims[1]);
        for (int i = 0; i < dims[0]; i++)
            for (int j = 0; j < dims[1]; j++)
                sampleTexture.SetPixel(i, j, new Color(imgData[i, j].r, imgData[i, j].g, imgData[i, j].b));
        sampleTexture.Apply();
        //byte[] bytes = sampleTexture.EncodeToPNG();
        byte[] bytes = sampleTexture.EncodeToPNG();
        System.IO.File.WriteAllBytes("Assets/Resources/" + fileName + ".png", bytes);
    }

    public static void WriteImage(string fileName, float[,] imgData, int[] dims, bool normalize = false)
    {
        if (normalize)
        {
            imgData = (float[,]) imgData.Clone();
            float max_val = FindMax(imgData);
            float min_val = FindMin(imgData);
            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++)
                    imgData[i, j] = (imgData[i, j] - min_val) / (max_val - min_val);
        }
        Texture2D sampleTexture = new Texture2D(dims[0], dims[1]);
        for (int i = 0; i < dims[0]; i++)
            for (int j = 0; j < dims[1]; j++)
                sampleTexture.SetPixel(i, j, new Color(imgData[i, j], imgData[i, j], imgData[i, j]));
        sampleTexture.Apply();
        //byte[] bytes = sampleTexture.EncodeToPNG();
        byte[] bytes = sampleTexture.EncodeToPNG();
        System.IO.File.WriteAllBytes("Assets/Resources/" + fileName + ".png", bytes);
    }

    public static void WriteImage(string fileName, Texture2D tex)
    {

        byte[] bytes = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes("Assets/Resources/" + fileName + ".png", bytes);
    }

    public static void ImportImageData(ref List<float[,,]> trainingInputData, ref List<float[]> trainingOutputData, List<Texture2D[]> imageDataset, bool grayscale)
    {
        int no_labels = 10;

        trainingInputData = new List<float[,,]>();
        trainingOutputData = new List<float[]>();
        int w = imageDataset[0][0].width;
        int h = imageDataset[0][0].height;

        for (int label = 0; label < imageDataset.Count; label++)
        {
            for (int n = 0; n < imageDataset[label].Length; n++)
            {
                float[,,] imageData = new float[1, 1, 1];
                ImportImageData(ref imageData, imageDataset[label][n], grayscale);

                float[] labels = new float[no_labels];
                labels[label] = 1;

                trainingOutputData.Add(labels);
                trainingInputData.Add(imageData);
            }
        }
    }
    public static void ImportImageData(ref float[,,] predictionInputData, Texture2D imageDataset, bool grayscale, bool normalize = true)
    {
        int w = imageDataset.width;
        int h = imageDataset.height;

        if (grayscale)
            predictionInputData = new float[1, w, h];
        else
            predictionInputData = new float[3, w, h];

        if (grayscale)
        {
            float maxPixel = 0;
            float[,,] imageData = new float[1, h, w];
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    Color pixel = imageDataset.GetPixel(i, j);
                    float sum = pixel.r + pixel.b + pixel.g;
                    if (sum > maxPixel)
                        maxPixel = sum;
                    imageData[0, i, j] = sum;
                }
            }
            if (normalize)
            {
                if (maxPixel > 0)
                    for (int i = 0; i < w; i++)
                        for (int j = 0; j < h; j++)
                            imageData[0, i, j] /= maxPixel;
            }
            predictionInputData = imageData;
        }
        else
        {
            float[] maxPixels = new float[3] { 0, 0, 0 };
            float[,,] imageData = new float[3, h, w];
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    Color pixel = imageDataset.GetPixel(i, j);
                    if (pixel.r > maxPixels[0])
                        maxPixels[0] = pixel.r;
                    if (pixel.g > maxPixels[1])
                        maxPixels[1] = pixel.g;
                    if (pixel.b > maxPixels[2])
                        maxPixels[2] = pixel.b;

                    imageData[0, i, j] = pixel.r;
                    imageData[1, i, j] = pixel.g;
                    imageData[2, i, j] = pixel.b;
                }
            }
            if (normalize)
            {
                for (int i = 0; i < w; i++)
                    for (int j = 0; j < h; j++)
                    {
                        if (maxPixels[0] > 0)
                            imageData[0, i, j] /= maxPixels[0];
                        if (maxPixels[1] > 0)
                            imageData[1, i, j] /= maxPixels[1];
                        if (maxPixels[2] > 0)
                            imageData[2, i, j] /= maxPixels[2];
                    }
            }
            predictionInputData = imageData;
        }
    }

    public static Texture2D Data2Texture(float [,,]imageData, int imgSize, bool monoChannel = false)
    {
        Texture2D outTex = new Texture2D(imgSize, imgSize);
        for (int i=0; i < imgSize; i++)
            for (int j=0; j < imgSize; j++)
            {
                if (monoChannel)
                    outTex.SetPixel(i, j, new Color(imageData[0, i, j], imageData[0, i, j], imageData[0, i, j]));
                else
                    outTex.SetPixel(i, j, new Color(imageData[0, i, j], imageData[1, i, j], imageData[2, i, j]));
            }
        outTex.Apply();
        return outTex;
    }

    public static Texture2D Data2Texture(float[] imageData, int imgSize, bool monoChannel = false)
    {
        Texture2D outTex = new Texture2D(imgSize, imgSize);
        for (int i = 0; i < imgSize; i++)
            for (int j = 0; j < imgSize; j++)
            {
                if (monoChannel)
                    outTex.SetPixel(i, j, new Color(imageData[i*imgSize + j], imageData[i * imgSize + j], imageData[i * imgSize + j]));
                else
                    outTex.SetPixel(i, j, new Color(imageData[i * imgSize + j], imageData[imgSize*imgSize +  i * imgSize + j], imageData[imgSize * imgSize + i * imgSize + j]));
            }
        outTex.Apply();
        return outTex;
    }
    public static Texture2D Data2Texture(float[,] imageData, int[] imgSize)
    {
        Texture2D outTex = new Texture2D(imgSize[0], imgSize[1]);

        for (int i = 0; i < imgSize[0]; i++)
            for (int j = 0; j < imgSize[1]; j++)
            {
                outTex.SetPixel(i, j, new Color(imageData[i, j], imageData[i, j], imageData[i, j]));
            }
        outTex.Apply();
        return outTex;
    }
    public static Texture2D Data2Texture(float[,,] imageData, int[] imgSize, bool monocolor = false)
    {
        bool useAlpha = false;
        if (((imageData.GetLength(0) == 4) & !monocolor) | ((imageData.GetLength(0) == 2) & monocolor))
            useAlpha = true;

        Texture2D outTex = new Texture2D(imgSize[0], imgSize[1]);
        for (int i = 0; i < imgSize[0]; i++)
            for (int j = 0; j < imgSize[1]; j++)
            {
                float alpha = 1;
                if (useAlpha)
                {
                    if (monocolor)
                        alpha = imageData[1, i, j];
                    else
                        alpha = imageData[3, i, j];
                }
                if (monocolor)
                    outTex.SetPixel(i, j, new Color(imageData[0, i, j], imageData[0, i, j], imageData[0, i, j], alpha));
                else
                    outTex.SetPixel(i, j, new Color(imageData[0, i, j], imageData[1, i, j], imageData[2, i, j], alpha));
            }
        outTex.Apply();
        return outTex;
    }

    public static void Data2Texture(ref Texture2D outTex, float[,,] imageData, int[] imgSize)
    {
        for (int i = 0; i < imgSize[0]; i++)
            for (int j = 0; j < imgSize[1]; j++)
            {
                outTex.SetPixel(i, j, new Color(imageData[0, i, j], imageData[1, i, j], imageData[2, i, j]));
            }
        outTex.Apply();
    }

    public static bool Is1DLayerType(Noedify.Layer layer)
    {
        switch (layer.layer_type)
        {
            case (Noedify.LayerType.FullyConnected):
            case (Noedify.LayerType.Input):
            case (Noedify.LayerType.ActivationFunction):
                return true;
            default: return false;
        }
    }

    public static bool Is2DLayerType(Noedify.Layer layer)
    {
        switch (layer.layer_type)
        {
            case (Noedify.LayerType.Input2D):
            case (Noedify.LayerType.Convolutional2D):
            case (Noedify.LayerType.TranspConvolutional2D):
            case (Noedify.LayerType.BatchNorm2D):
            case (Noedify.LayerType.Pool2D):
                return true;
            default: return false;
        }
    }

    public static bool Is3DLayerType(Noedify.Layer layer)
    {
        switch (layer.layer_type)
        {
            case (Noedify.LayerType.Input3D):
            case (Noedify.LayerType.Convolutional3D):
            case (Noedify.LayerType.TranspConvolutional3D):
            case (Noedify.LayerType.BatchNorm3D):
                return true;
            default: return false;
        }
    }

    public static bool IsInputLayerType(Noedify.Layer layer)
    {
        switch (layer.layer_type)
        {
            case (Noedify.LayerType.Input):
            case (Noedify.LayerType.Input2D):
            case (Noedify.LayerType.Input3D):
                return true;
            default: return false;
        }
    }

    public static int FilterIndex(int ix, int iy, int input_ch, Noedify.Layer layerInfo)
    {
        return input_ch * layerInfo.conv2DLayer.filterSize[0] * layerInfo.conv2DLayer.filterSize[1] + iy * layerInfo.conv2DLayer.filterSize[0] + ix;
    }

    public static bool CompareArray<T>(T[] array1, T[] array2)
    {
        if (array1.Length != array2.Length)
        {
            Debug.Log("CompareArray: Array size mismatch (" + array1.Length + ") != (" + array2.Length + ")");
            return false;
        }
        if (array1 == null && array2 == null)
        {
            return true;
        }

        if ((array1 == null) || (array2 == null))
        {
            return false;
        }

        if (array1.Length != array2.Length)
        {
            return false;
        }
        if (array1.Equals(array2))
        {
            return true;
        }
        else
        {
            for (int Index = 0; Index < array1.Length; Index++)
            {
                if (!Equals(array1[Index], array2[Index]))
                {
                    return false;
                }
            }
        }
        return true;
    }

    public static bool CompareArray<T>(T[,] array1, T[,] array2, string name = "array")
    {
        if (array1.GetLength(0) != array2.GetLength(0) | array1.GetLength(1) != array2.GetLength(1))
        {
            Debug.Log("CompareArray: " + name + " size mismatch (" + array1.GetLength(0) + "," + array1.GetLength(1) + ") != (" + array2.GetLength(0) + "," + array2.GetLength(1) + ")");
            return false;
        }
        if (array1 == null && array2 == null)
        {
            return true;
        }

        if ((array1 == null) || (array2 == null))
        {
            return false;
        }

        if (array1.Length != array2.Length)
        {
            return false;
        }
        if (array1.Equals(array2))
        {
            return true;
        }
        else
        {
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                for (int j = 0; j < array1.GetLength(1); j++)
                {
                    if (!Equals(array1[i,j], array2[i,j]))
                    {
                        Debug.Log(name + " array mismatch [" + i + "," + j + "]: " + array1[i, j] + " != " + array2[i, j]);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public static List<Noedify.InputArray> ConvertInputList(List<float[,,]> inputFloatArrayList)
    {
        List<Noedify.InputArray> newList = new List<Noedify.InputArray>();
        for (int n = 0; n < inputFloatArrayList.Count; n++)
        {
            newList.Add(new Noedify.InputArray(inputFloatArrayList[n]));
        }
        return newList;
    }

    public static Vector3 FindMax(Vector3[] array)
    {
        /*
         * Find max value of 1D Vector3 array
        */
        Vector3 max_vert = Vector3.zero;
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i].x > max_vert.x)
                max_vert.x = array[i].x;
            if (array[i].y > max_vert.y)
                max_vert.y = array[i].y;
            if (array[i].z > max_vert.z)
                max_vert.z = array[i].z;
        }
        return max_vert;
    }

    public static Vector3 FindMax(Vector3[] array, int[] sampleList)
    {
        /*
         * Find max value of 1D Vector3 array
        */
        Vector3 max_vert = Vector3.zero;
        for (int i = 0; i < sampleList.Length; i++)
        {
            int n = sampleList[i];
            if (array[n].x > max_vert.x)
                max_vert.x = array[n].x;
            if (array[n].y > max_vert.y)
                max_vert.y = array[n].y;
            if (array[n].z > max_vert.z)
                max_vert.z = array[n].z;
        }
        return max_vert;
    }

    public static float FindMax(float[,] array)
    {
        /*
         * Find max value of 1D Vector3 array
        */
        float max_vert = -1;
        for (int i = 0; i < array.GetLength(0); i++)
        {
            for (int j = 0; j < array.GetLength(1); j++)
            {
                if (array[i,j] > max_vert)
                    max_vert = array[i, j];
            }
        }
        return max_vert;
    }

    public static Vector2 FindMax(Vector2[] array)
    {
        /*
         * Find max value of 1D Vector2 array
        */
        Vector3 max_vert = Vector2.zero;
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i].x > max_vert.x)
                max_vert.x = array[i].x;
            if (array[i].y > max_vert.y)
                max_vert.y = array[i].y;
        }
        return max_vert;
    }

    public static float FindMax(float[] array)
    {
        /*
         * Find max value of 1D float array
        */
        float max = 0;
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i] > max)
                max = array[i];
        }
        return max;
    }

    public static float FindMax(float[,,,] array, int ch = -1)
    {
        /*
         * Find max value of 4D float array 
         * if ch>-1: search only along channel 'ch'
         * otherwise, across all channels
         */
        int c = array.GetLength(0);
        int d = array.GetLength(3);
        int h = array.GetLength(2);
        int w = array.GetLength(1);

        int ch_start = 0;
        int ch_end = ch_start + 1;
        if (ch > -1)
        {
            ch_start = ch;
            ch_end = ch + 1;
        }
        float max = 0;
        for (int ic = ch_start; ic < ch_end; ic++)
            for (int iw = 0; iw < w; iw++)
                for (int ih = 0; ih < h; ih++)
                    for (int id = 0; id < d; id++)
                    {
                        if (array[ic, iw, ih, id] > max)
                            max = array[ic, iw, ih, id];
                    }
        return max;
    }

    public static Vector3 FindMin(Vector3[] array)
    {
        Vector3 min_vert = new Vector3(10000, 10000, 10000);
        int y_min = -1;
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i].x <= min_vert.x)
                min_vert.x = array[i].x;
            if (array[i].y <= min_vert.y)
            {
                min_vert.y = array[i].y;
                y_min = i;
            }
            if (array[i].z <= min_vert.z)
                min_vert.z = array[i].z;
        }
        //print("ymin: " + y_min);
        return min_vert;
    }

    public static Vector3 FindMin(Vector3[] array, int[] sampleList)
    {
        Vector3 min_vert = new Vector3(10000, 10000, 10000);
        int y_min = -1;
        for (int i = 0; i < sampleList.Length; i++)
        {
            int n = sampleList[i];
            if (array[n].x <= min_vert.x)
                min_vert.x = array[n].x;
            if (array[n].y <= min_vert.y)
            {
                min_vert.y = array[n].y;
                y_min = n;
            }
            if (array[n].z <= min_vert.z)
                min_vert.z = array[n].z;
        }
        //print("ymin: " + y_min);
        return min_vert;
    }

    public static float FindMin(float[,] array)
    {
        /*
         * Find max value of 1D Vector3 array
        */
        float min_vert = 1000;
        for (int i = 0; i < array.GetLength(0); i++)
        {
            for (int j = 0; j < array.GetLength(1); j++)
            {
                if (array[i, j] < min_vert)
                    min_vert = array[i, j];
            }
        }
        return min_vert;
    }

    public static Vector2 FindMin(Vector2[] array)
    {
        Vector2 min_vert = new Vector2(10000, 10000);
        int y_min = -1;
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i].x <= min_vert.x)
                min_vert.x = array[i].x;
            if (array[i].y <= min_vert.y)
            {
                min_vert.y = array[i].y;
                y_min = i;
            }
        }
        return min_vert;
    }

    public static float FindMin(float[] array)
    {
        float min = 999;
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i] < min)
                min = array[i];
        }
        return min;
    }

    public static float FindMin(float[,,,] array, int ch = -1)
    {
        /*
         * Find min value of 4D float array 
         * if ch>-1: search only along channel 'ch'
         * otherwise, across all channels
         */

        int c = array.GetLength(0);
        int d = array.GetLength(3);
        int h = array.GetLength(2);
        int w = array.GetLength(1);

        int ch_start = 0;
        int ch_end = ch_start + 1;
        if (ch > -1)
        {
            ch_start = ch;
            ch_end = ch + 1;
        }
        float min = 0;
        for (int ic = ch_start; ic < ch_end; ic++)
            for (int iw = 0; iw < w; iw++)
                for (int ih = 0; ih < h; ih++)
                    for (int id = 0; id < d; id++)
                    {
                        if (array[ic, iw, ih, id] < min)
                            min = array[ic, iw, ih, id];
                    }
        return min;
    }

    public static Vector3 FindMean(Vector3[] array)
    {
        /*
         * Find max value of 1D Vector3 array
        */
        Vector3 mean_vert = Vector3.zero;
        for (int i = 0; i < array.Length; i++)
        {
            mean_vert += array[i];
        }
        return mean_vert / array.Length;
    }

#if NOEDIFY_NORELEASE

    public static float[,,] DebugInput(Vector3Int arraySize, string arrayType)
    {
        float[,,] testArray = new float[arraySize.x, arraySize.y, arraySize.z];

        if (arrayType == "int_incremental")
        {
            for (int iz=0; iz < arraySize.z; iz++) 
                for (int iy=0; iy < arraySize.y; iy++)
                    for (int ix=0; ix < arraySize.x; ix++)
                        testArray[ix, iy, iz] = iz * arraySize.y * arraySize.x + iy * arraySize.x + ix;
                    
        }
        else if (arrayType == "clamped_incremental")
        {
            for (int iz = 0; iz < arraySize.z; iz++)
                for (int iy = 0; iy < arraySize.y; iy++)
                    for (int ix = 0; ix < arraySize.x; ix++)
                        testArray[ix, iy, iz] = ((float)(iz * arraySize.y * arraySize.x + iy * arraySize.x + ix)) / ((float)arraySize.x * arraySize.y * arraySize.z);

        }

        return testArray;
    }

    public static float[,,] CombineArrays(float[,] a, float[,] b, float[,] c, int[] size)
    {
        float[,,] output = new float[3, size[1], size[0]];

        for (int i=0; i < size[0]; i++)
            for (int j=0; j < size[1]; j++)
            {
                output[0, i, j] = a[i, j];
                output[1, i, j] = b[i, j];
                output[2, i, j] = c[i, j];
            }

        return output;
    }

    public static void PrintByte(byte newByte)
    {
        newByte.ToString();
    }

    public static void rgb2ycbcr(float[,,] arrayrgb, ref float[,] Y, ref float[,] Cb, ref float[,] Cr, int textureSize)
    {
        for (int i = 0; i < textureSize; i++)
        {
            for (int j = 0; j < textureSize; j++)
            {
                float[] rgb = new float[3] { arrayrgb[0, i, j], arrayrgb[1, i, j], arrayrgb[2, i, j] };
                float[] ycbcr = rgb2ycbcr_pixel(rgb);
                Y[i, j] = ycbcr[0];
                Cb[i, j] = ycbcr[1];
                Cr[i, j] = ycbcr[2];
                //Debug.Log("(" + Cb[i, j] + ", " + Cr[i,j] + ")");
            }
        }
    }

    public static void rgb2ycbcr(Texture2D input_texture, ref float[,] Y, ref float[,] Cb, ref float[,] Cr)
    {
        int w = input_texture.width;
        int h = input_texture.height;
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                Color pixel = input_texture.GetPixel(i, j);
                float[] rgb = new float[3] { pixel.r, pixel.g, pixel.b };
                float[] ycbcr = rgb2ycbcr_pixel(rgb);
                Y[i, j] = ycbcr[0];
                Cb[i, j] = ycbcr[1];
                Cr[i, j] = ycbcr[2];
                //Debug.Log("(" + Cb[i, j] + ", " + Cr[i,j] + ")");
            }
        }
    }

    public static void rgb2ycbcr(Texture2D input_texture, ref float[,] Y, ref float[,] Cb, ref float[,] Cr, ref float[,] alpha)
    {
        int w = input_texture.width;
        int h = input_texture.height;
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                Color pixel = input_texture.GetPixel(i, j);
                float[] rgb = new float[3] { pixel.r, pixel.g, pixel.b };
                float[] ycbcr = rgb2ycbcr_pixel(rgb);
                Y[i, j] = ycbcr[0];
                Cb[i, j] = ycbcr[1];
                Cr[i, j] = ycbcr[2];
                alpha[i, j] = pixel.a;
                //Debug.Log("(" + Cb[i, j] + ", " + Cr[i,j] + ")");
            }
        }
    }

    private static float[] rgb2ycbcr_pixel(float[] rgb)
    {
        float Y = (float)(0.29803922f * rgb[0] + 0.58431373f * rgb[1] + 0.11372549f * rgb[2]);
        float Cb = (float)(0.5f - 0.170588f * rgb[0] - 0.33137255f * rgb[1] + 0.5f * rgb[2]);
        float Cr = (float)(0.5f + 0.5f * rgb[0] - 0.41764706f * rgb[1] - 0.0804f * rgb[2]);
        return new float[3] { Y, Cb, Cr };
    }

    public static void ycbcr2rgb(float[,] Y, float[,] Cb, float[,] Cr, ref float[,,] arrayrgb, int[] textureSize)
    {
        for (int i = 0; i < textureSize[0]; i++)
        {
            for (int j = 0; j < textureSize[1]; j++)
            {
                float[] ycbcr = new float[3] { Y[i,j], Cb[i,j], Cr[i,j] };
                float[] rgb = ycbcr2rgb_pixel(ycbcr);
                arrayrgb[0, i, j] = rgb[0];
                arrayrgb[1, i, j] = rgb[1];
                arrayrgb[2, i, j] = rgb[2];
            }
        }
    }

    public static void ycbcr2rgb(float[,,] arrayYCbCr, ref float[,,] arrayrgb, int[] textureSize)
    {
        bool useAlpha = false;
        if (arrayYCbCr.GetLength(0) == 4)
        {
            useAlpha = true;
        }

        for (int i = 0; i < textureSize[0]; i++)
        {
            for (int j = 0; j < textureSize[1]; j++)
            {
                float alpha = 1;
                if (useAlpha)
                    alpha = arrayYCbCr[3, i, j];
                float[] ycbcr = new float[3] { arrayYCbCr[0,i,j], arrayYCbCr[1,i,j], arrayYCbCr[2,i,j] };
                float[] rgb = ycbcr2rgb_pixel(ycbcr);
                arrayrgb[0, i, j] = rgb[0];
                arrayrgb[1, i, j] = rgb[1];
                arrayrgb[2, i, j] = rgb[2];
                arrayrgb[3, i, j] = alpha;
            }
        }
    }

    private static float[] ycbcr2rgb_pixel(float[] YCrCb)
    {
        float R = YCrCb[0] + (YCrCb[2] - 0.5f) * 1.40200f;
        float G = YCrCb[0] + (YCrCb[1] - 0.5f) * -0.34414f + (YCrCb[2] - 0.5f) * -0.71414f;
        float B = YCrCb[0] + (YCrCb[1] - 0.5f) * 1.77200f;
        return new float[3] { R, G, B };
    }



#endif

}
