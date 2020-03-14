using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet
{
    class OutputData
    {
        [ColumnName("PredictedLabel")]
        public string Species { get; set; }

        [ColumnName("Score")]
        public float[] Scores { get; set; }
    }
}
