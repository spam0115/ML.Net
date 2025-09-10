// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TensorFlow
{
    public static class ArrayExtensions
    {
        public static int[] ToIntArray(this long[] source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            int[] result = new int[source.Length];
            for (int i = 0; i < source.Length; i++)
            {
                if (source[i] > int.MaxValue || source[i] < int.MinValue)
                    throw new OverflowException($"Value at index {i} ({source[i]}) is out of range for Int32.");
                result[i] = (int)source[i];
            }
            return result;
        }
    }
}
