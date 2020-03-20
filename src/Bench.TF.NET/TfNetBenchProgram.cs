using System;
using TensorFlowNET.Examples;

namespace Bench.TF.NET
{
    class TfNetBenchProgram
    {
        static void Main()
        {
            var sample = new DigitRecognitionCNN {
                Config = new ExampleConfig {
                    Name = "DigitRecognitionCNN",
                }
            };
            sample.Run();
        }
    }
}
