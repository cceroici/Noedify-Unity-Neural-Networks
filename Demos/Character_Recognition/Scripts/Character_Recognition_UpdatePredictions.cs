using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Character_Recognition_UpdatePredictions : MonoBehaviour
{
    public float maxFontSize = 150;
    public float minFontSize = 10;

    public bool plotNow = false;

    public Text[] text_pred;
    public float[] plotValues;

    string[] symbols = new string[] {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};



    // Start is called before the first frame update
    void Start()
    {
        plotValues = new float[10];
        UpdatePred(plotValues);
    }

    // Update is called once per frame
    void Update()
    {
        if (plotNow)
        {
            plotNow = false;
            UpdatePred(plotValues);
        }
    }

    public void UpdatePred(float[] values)
    {
        float[] predictions = (float[]) values.Clone();

        float max = Noedify_Utils.FindMax(predictions);
        float min = Noedify_Utils.FindMin(predictions);
        for (int i=0; i < predictions.Length; i++)
            predictions[i] = (predictions[i] - min)/(max-min);

        for (int i=0; i < text_pred.Length; i++)
        {
            int max_pred_idx = Noedify_Utils.ArgMax(predictions);
            string pred_str = symbols[max_pred_idx];
            text_pred[i].fontSize = (int)Mathf.Lerp(minFontSize, maxFontSize, predictions[max_pred_idx]);
            text_pred[i].text = pred_str;
            predictions[max_pred_idx] = -1000;
        }

    }
}
