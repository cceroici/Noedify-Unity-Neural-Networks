using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Character_Recognition_DrawController : MonoBehaviour
{
    public RenderTexture renderTexture;
    public Material sampleMat;

    private void Start()
    {
        SampleDrawing(new int[2] { 64, 64 });
    }

    // This will retrive a 2D float representing the pixel data from a screenshot of the main camera
    public float[,] SampleDrawing(int[] dim)
    {
        float[,] outputData = new float[dim[0], dim[1]];

        Texture2D tex = new Texture2D(dim[0], dim[1], TextureFormat.RGB24, false);
        RenderTexture rt = new RenderTexture(dim[0], dim[1], 24);
        Camera.main.targetTexture = rt;
        Camera.main.Render();
        RenderTexture.active = rt;
        Rect rectReadPixels = new Rect(0, 0, dim[0], dim[1]);
        tex.ReadPixels(rectReadPixels, 0, 0);
        tex.Apply();

        Camera.main.targetTexture = null;

        for (int j = 0; j < tex.height; j++)
        {
            for (int i = 0; i < tex.width; i++)
            {
                Color pixelColor = tex.GetPixel(i, j);
                float grayscale = (pixelColor.r + pixelColor.g + pixelColor.b) / 3f;
                //grayscale = 0.0f;
                outputData[i, j] = grayscale;
            }
        }
        
        Noedify_Utils.WriteImage("sample_letter.png", outputData, new int[2] {28,28}, true);
        outputData = Noedify_Utils.RotateImage90(outputData);
        outputData = Noedify_Utils.FlipLR(outputData);
        //outputData = Noedify_Utils.FlipUD(outputData);

        sampleMat.SetTexture("_MainTex", tex);
        Debug.Log("max sig: " + Noedify_Utils.FindMax(outputData));
        return outputData;
    }



}
