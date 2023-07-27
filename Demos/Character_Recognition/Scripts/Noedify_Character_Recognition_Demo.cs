using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Noedify_Character_Recognition_Demo : MonoBehaviour
{
    public Character_Recognition_DrawController drawController;
    public bool modelImportComplete = false;
    public Noedify.Net detectionModel;

    public Character_Recognition_UpdatePredictions showPredictions;
    
    public Noedify_Solver.SolverMethod processor = Noedify_Solver.SolverMethod.CPU;
    public ComputeShader _shader;
    Noedify_Solver solver;
    const string modelCode = "2_28,28,1_bcbcbaa_6,6,4,2,2,0,0_1e-05_4,4,6,2,2,0,0_1e-05_3,3,10,2,2,0,0_128_26_LRLRRRT";

    /*
    string[] symbols = new string[] {"0", "1", "2", "3", "4", "5", "6", "7", "8" , "9",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    */
    string[] symbols = new string[] {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};


    string model_name = "CharRecog";
    string parameters_file = "CharRecog";

    void Start()
    {
        detectionModel = BuildAndImportModel();

        float [,,] test_input_array = new float[1,28,28];
        test_input_array = Noedify_Utils.AddSingularDim(Noedify_Utils.Ones(28,28));
        //test_input_array[0,0,0] = 10f;
        /*
        float[,] drawing = drawController.SampleDrawing(new int[2] { 28, 28 });

        float[,,] character_input_array = Noedify_Utils.AddSingularDim(drawing);

        Noedify.InputArray test_input = new Noedify.InputArray(character_input_array);
        //Noedify.InputArray test_input = new Noedify.InputArray(test_input_array);
            
        solver.Evaluate(detectionModel, test_input, processor);
        
        Noedify_Utils.PrintArray(solver.prediction, "pred");
        print("Predicted class: " + Noedify_Utils.ArgMax(solver.prediction) + " symbol: " + symbols[Noedify_Utils.ArgMax(solver.prediction)]);
        */
        StartCoroutine(Predict());
    }


    
    Noedify.Net BuildAndImportModel()
    {
        modelImportComplete = false;
        int no_labels = 10;

        bool importTensorflowModel = false;

        Noedify.Net model = new Noedify.Net();


        // Attempt to load network saved as a binary file
        // This is much faster than importing form a parameters file
        if (!(Noedify_Manager.LoadCompressedModel(ref model, model_name, "")))
        {
            { // Load model configuration
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();
                if (!Noedify_Manager.LoadModelConfiguration(ref model, modelCode))
                    Debug.Log("model load from code failed.");
                sw.Stop();
                Debug.Log("***Model load config time: " + sw.ElapsedMilliseconds + " ms");
            }
            { // Build model
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();
                model.BuildNetwork(true);
                sw.Stop();
                Debug.Log("***Network build time: " + sw.ElapsedMilliseconds + " ms");
                Debug.Log("total # weights: " + model.total_no_weights);
            }
            { // Load network parameters (compilable)
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Reset(); sw.Start();
                bool status = Noedify_Manager.ImportNetworkParameters(model, parameters_file);
                //bool status = false;
                if (status)
                    Debug.Log("load successful");
                else
                    Debug.Log("load failed");
                sw.Stop();
                Debug.Log("***Network parameter load time: " + sw.ElapsedMilliseconds + " ms");
            }
            /*
            { // Load network parameters (compilable)
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Reset(); sw.Start();
                bool status = Noedify_Manager.ImportNetworkCompilableParameters(model);
                //bool status = false;
                if (status)
                    Debug.Log("load successful");
                else
                    Debug.Log("load failed");
                sw.Stop();
                Debug.Log("***Network parameter load time: " + sw.ElapsedMilliseconds + " ms");
            }
            */
            Noedify_Manager.SavedCompressedModel compressedModel = new Noedify_Manager.SavedCompressedModel(model, "Faceblend_", System.DateTime.Today);
            compressedModel.Save();
        }
        else
        {
            Debug.Log("Loaded Compressed model file");
        }
        // If the binary file doesn't exist yet, import the parameters from the Tensorflow file
        // This is slower, so we will save the network as a binary file after importing it
        if (importTensorflowModel)
        {
            bool status;
            /* Input layer */
            Noedify.Layer inputLayer = new Noedify.Layer(
                Noedify.LayerType.Input2D, // layer type
                new int[2] { 28, 28 }, // input size
                1, // # of channels
                "input layer" // layer name
                );
            detectionModel.AddLayer(inputLayer);

            // Hidden layer 0
            Noedify.Layer hiddenLayer0 = new Noedify.Layer(
                Noedify.LayerType.FullyConnected, // layer type
                600, // layer size
                Noedify.ActivationFunction.Sigmoid, // activation function
                "fully connected 1" // layer name
                );
            detectionModel.AddLayer(hiddenLayer0);

            // Hidden layer 2
            Noedify.Layer hiddenLayer1 = new Noedify.Layer(
                Noedify.LayerType.FullyConnected, // layer type
                300, // layer size
                Noedify.ActivationFunction.Sigmoid, // activation function
                "fully connected 2" // layer name
                );
            detectionModel.AddLayer(hiddenLayer1);

            // Hidden layer 2
            Noedify.Layer hiddenLayer2 = new Noedify.Layer(
                Noedify.LayerType.FullyConnected, // layer type
                140, // layer size
                Noedify.ActivationFunction.Sigmoid, // activation function
                "fully connected 3" // layer name
                );
            detectionModel.AddLayer(hiddenLayer2);

            /* Output layer */
            Noedify.Layer outputLayer = new Noedify.Layer(
                Noedify.LayerType.Output, // layer type
                no_labels, // layer size
                Noedify.ActivationFunction.Sigmoid, // activation function
                "output layer" // layer name
                );
            detectionModel.AddLayer(outputLayer);

            detectionModel.BuildNetwork();

            status = Noedify_Manager.ImportNetworkParameters(detectionModel, "FC_mnist_600x300x140_parameters");
            if (status)
                 Debug.Log("Successfully loaded model.");
            else
            {
                Debug.Log("Tensorflow model load failed. Have you moved the \"...Assets/Noedify/Resources\" folder to: \"...Assets/Resources\" ?");
                Debug.Log("All model parameter files must be stored in: \"...Assets/Resources/Noedify/ModelParameterFiles\"");
                return null;
            }
            detectionModel.SaveModel("Noedify-Model_Digit_Drawing_Test");
            Debug.Log("Saved binary model file \"Noedify-Model_Digit_Drawing_Test\"");
        }
        solver = Noedify.CreateSolver();

        solver.suppressMessages = false;
        modelImportComplete = true;
        model._shader = _shader;
        return model;
    }

    public static void LoadComputationModel(ref Noedify.Net model, ref string loadedModel, string override_modelCode = "")
    {
        model = new Noedify.Net();
        if (!(Noedify_Manager.LoadCompressedModel(ref model, "Faceblend_", "")))
        {
            string model_code;
            if (override_modelCode == "")
                model_code = Noedify_CompileableNetParams.model_code;
            else
                model_code = override_modelCode;
            { // Load model configuration
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();
                if (!Noedify_Manager.LoadModelConfiguration(ref model, model_code))
                    Debug.Log("model load from code failed.");
                sw.Stop();
                Debug.Log("***Model load config time: " + sw.ElapsedMilliseconds + " ms");
            }
            { // Build model
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();
                model.BuildNetwork(true);
                sw.Stop();
                Debug.Log("***Network build time: " + sw.ElapsedMilliseconds + " ms");
                Debug.Log("total # weights: " + model.total_no_weights);
            }
            { // Load network parameters
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Reset(); sw.Start();
                bool status = Noedify_Manager.ImportNetworkCompilableParameters(model);
                //bool status = false;
                if (status)
                    Debug.Log("load successful");
                else
                    Debug.Log("load failed");
                sw.Stop();
                Debug.Log("***Network parameter load time: " + sw.ElapsedMilliseconds + " ms");
            }
            Noedify_Manager.SavedCompressedModel compressedModel = new Noedify_Manager.SavedCompressedModel(model, "Faceblend_", System.DateTime.Today);
            compressedModel.Save();
        }
        else
        {
            Debug.Log("Loaded Compressed model file");
        }
    }
/*
    public static void Predict_Expression(Noedify.Net model, ref Noedify_Solver solver, FaceBlendProfile bp, ref Vector3[] deltaVerts, GameObject headObj, Mesh mesh)
    {
        ComputeShader _FB_shader = Resources.Load<ComputeShader>("FaceBlend/ComputeShader/FaceBlend_Compute");
        model._shader = Resources.Load<ComputeShader>("Noedify_Compute");

        float[,,,] input3D = Noedify_Utils.AddSingularDim(bp.dm);
        //Noedify.InputArray model_input = new Noedify.InputArray(input3D);

        Noedify.InputArray model_input = new Noedify.InputArray(Noedify_Utils.AddSingularDim(
            UnpackHeadData(bp.head_data, bp.dm_gridSize)));

        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();
        solver.Evaluate(model, model_input, Noedify_Solver.SolverMethod.GPU);
        sw.Stop();

        float[,,,] morph_map = new float[3, bp.dm_gridSize.x, bp.dm_gridSize.y, bp.dm_gridSize.z];
        for (int ix = 0; ix < bp.dm_gridSize.x; ix++)
            for (int iy = 0; iy < bp.dm_gridSize.y; iy++)
                for (int iz = 0; iz < bp.dm_gridSize.z; iz++)
                {
                    morph_map[0, ix, iy, iz] = solver.prediction[iz * bp.dm_gridSize.y * bp.dm_gridSize.x + iy * bp.dm_gridSize.x + ix];
                    morph_map[1, ix, iy, iz] = solver.prediction[bp.dm_gridSize.z * bp.dm_gridSize.y * bp.dm_gridSize.x +
                        iz * bp.dm_gridSize.y * bp.dm_gridSize.x + iy * bp.dm_gridSize.x + ix];
                    morph_map[2, ix, iy, iz] = solver.prediction[2 * bp.dm_gridSize.z * bp.dm_gridSize.y * bp.dm_gridSize.x +
                        iz * bp.dm_gridSize.y * bp.dm_gridSize.x + iy * bp.dm_gridSize.x + ix];
                }

        Vector3[] verts_delta = FaceBlend_Utils.Apply_morphMap_GPU(bp.x_dm, bp.y_dm, bp.z_dm, bp.dm_gridSize, bp.verts_normalized_base, morph_map, _FB_shader);

        Vector3 maxV = Noedify_Utils.FindMax(verts_delta);
        Debug.Log("Max delta (norm): (" + maxV.x + ", " + maxV.y + ", " + maxV.z + ")");
        verts_delta = FaceBlend_Utils.DenormalizeArray(verts_delta, bp.vertLims[0], bp.vertLims[1], false);

        deltaVerts = verts_delta;

        maxV = Noedify_Utils.FindMax(verts_delta);
        Debug.Log("Max delta (denorm): (" + maxV.x + ", " + maxV.y + ", " + maxV.z + ")");

    }
*/

    public static bool ModelIsInitialized(Noedify.Net model)
    {
        if (model == null)
            return false;
        if (model.LayerCount() == 0)
            return false;
        else if (model.layers[1].conv3DLayer.connections == null)
            return false;
        return true;

    }


    IEnumerator Predict()
    {
        while (true)
        {
            float[,,] test_input_array = new float[1, 28, 28];
            //test_input_array[0, 0, 0] = 1;

            float[,] drawing = drawController.SampleDrawing(new int[2] { 28, 28 });

            float[,,] character_input_array = Noedify_Utils.AddSingularDim(drawing);

            test_input_array = Noedify_Utils.AddSingularDim(Noedify_Utils.Ones(28,28));

            Noedify.InputArray test_input = new Noedify.InputArray(character_input_array);

            solver.Evaluate(detectionModel, test_input, processor);

            Noedify_Utils.PrintArray(solver.prediction, "pred");
            print("Predicted class: " + Noedify_Utils.ArgMax(solver.prediction) + " symbol: " + symbols[Noedify_Utils.ArgMax(solver.prediction)]);
            showPredictions.UpdatePred(solver.prediction);
            yield return new WaitForSeconds(.1f);
        }
    }

}
