﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet
{
    class InputData
    {
        [LoadColumn(0)]
        public float SepalLength { get; set; }

        [LoadColumn(1)]
        public float SepalWidth { get; set; }

        [LoadColumn(2)]
        public float PetalLength { get; set; }

        [LoadColumn(3)]
        public float PetalWidth { get; set; }

        [LoadColumn(4)]
        public string Species { get; set; }
    }
}
