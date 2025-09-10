// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NumSharp;
using Tensorflow;

namespace ShapeInterop
{
    // How to treat unknown (-1) or missing dimensions
    public enum UnknownDimPolicy
    {
        Throw,        // error on any unknown dimension
        UseMinusOne,  // keep unknown as -1 (metadata-friendly; cannot allocate NDArray)
        UseOne        // replace unknown with 1 (convenient for concrete shapes)
    }

    public static class TfTensorFlowShapeToNumSharpShapeToNumSharp
    {
        // Core conversion from TF dims (nullable long: null => unknown) to NumSharp.Shape
        public static NumSharp.Shape ToNumSharpShapeFromDims(IEnumerable<long?> tfDims,
                                                    UnknownDimPolicy policy = UnknownDimPolicy.UseMinusOne)
        {
            if (tfDims == null) throw new ArgumentNullException(nameof(tfDims));

            var list = tfDims.ToList();

            // Scalar: empty shape
            if (list.Count == 0)
                return new NumSharp.Shape(Array.Empty<int>());

            var result = new int[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                var d = list[i];
                if (d == null || d < 0)
                {
                    switch (policy)
                    {
                        case UnknownDimPolicy.Throw:
                            throw new InvalidOperationException($"Unknown TF dim at index {i} cannot be converted.");
                        case UnknownDimPolicy.UseMinusOne:
                            result[i] = -1;
                            break;
                        case UnknownDimPolicy.UseOne:
                            result[i] = 1;
                            break;
                    }
                }
                else
                {
                    if (d > int.MaxValue)
                        throw new OverflowException($"TF dim {d} at index {i} exceeds Int32.MaxValue.");
                    result[i] = (int)d.Value;
                }
            }
            return new NumSharp.Shape(result);
        }
    }

    public static class TensorFlowNetExtensions
    {
        public static NumSharp.Shape ToNumSharpShape(this Tensorflow.Shape tfShape,
                                            UnknownDimPolicy policy = UnknownDimPolicy.UseMinusOne)
        {
            if (tfShape == null) throw new ArgumentNullException(nameof(tfShape));

            // 1) Try common direct members first (dims / as_list)
            var dims = TryGetDimsVector(tfShape);
            if (dims != null)
                return TfTensorFlowShapeToNumSharpShapeToNumSharp.ToNumSharpShapeFromDims(dims, policy);

            // 2) Fallback: rank/ndim + indexer get_Item(int)
            var rank = TryGetRank(tfShape);
            if (rank == null || rank < 0)
            {
                // unknown rank
                switch (policy)
                {
                    case UnknownDimPolicy.Throw:
                        throw new InvalidOperationException("TensorFlow shape has unknown rank.");
                    case UnknownDimPolicy.UseMinusOne:
                        return new NumSharp.Shape(-1);
                    case UnknownDimPolicy.UseOne:
                        return new NumSharp.Shape(1);
                }
            }

            var getItem = tfShape.GetType().GetMethod("get_Item", new[] { typeof(int) });
            if (getItem == null)
                throw new NotSupportedException("TensorFlow.Tensorflow.Shape indexer not found.");

            IEnumerable<long?> seq = Enumerable.Range(0, rank.Value).Select(i =>
            {
                var val = getItem.Invoke(tfShape, new object[] { i });
                long v = Convert.ToInt64(val);
                return v >= 0 ? (long?)v : null;
            });

            return TfTensorFlowShapeToNumSharpShapeToNumSharp.ToNumSharpShapeFromDims(seq, policy);
        }

        // Try to extract dims via 'dims' field/property or 'as_list()'
        private static IEnumerable<long?> TryGetDimsVector(Tensorflow.Shape shape)
        {
            var t = shape.GetType();

            // Try property 'dims'
            var dimsProp = t.GetProperty("dims", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (dimsProp != null)
            {
                var obj = dimsProp.GetValue(shape);
                var converted = ConvertDimsArrayToNullableLongs(obj);
                if (converted != null) return converted;
            }

            // Try field 'dims'
            var dimsField = t.GetField("dims", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (dimsField != null)
            {
                var obj = dimsField.GetValue(shape);
                var converted = ConvertDimsArrayToNullableLongs(obj);
                if (converted != null) return converted;
            }

            // Try method 'as_list()' (common in TF.NET)
            var asList = t.GetMethod("as_list", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (asList == null)
                asList = t.GetMethod("AsList", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            if (asList != null)
            {
                var obj = asList.GetParameters().Length == 0 ? asList.Invoke(shape, null) : null;
                var converted = ConvertDimsArrayToNullableLongs(obj);
                if (converted != null) return converted;
            }

            return null;
        }

        // Try to get rank via 'rank' or 'ndim'
        private static int? TryGetRank(Tensorflow.Shape shape)
        {
            var t = shape.GetType();
            int? getProp(string name)
            {
                var p = t.GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (p == null) return null;
                var v = p.GetValue(shape);
                return v == null ? (int?)null : Convert.ToInt32(v);
            }
            return getProp("rank") ?? getProp("ndim");
        }

        // Convert any supported dims array into IEnumerable<long?> with null for unknown
        private static IEnumerable<long?> ConvertDimsArrayToNullableLongs(object dimsArray)
        {
            if (dimsArray == null) return null;

            switch (dimsArray)
            {
                case long?[] a:
                    return a; // already nullable
                case long[] a:
                    return a.Select(x => x >= 0 ? (long?)x : null);
                case int[] a:
                    return a.Select(x => x >= 0 ? (long?)x : null);
                case IEnumerable<long> e:
                    return e.Select(x => x >= 0 ? (long?)x : null);
                case IEnumerable<int> e:
                    return e.Select(x => x >= 0 ? (long?)x : null);
                case Array arr:
                    {
                        var list = new List<long?>(arr.Length);
                        foreach (var o in arr)
                        {
                            if (o == null) { list.Add(null); continue; }
                            long v = Convert.ToInt64(o);
                            list.Add(v >= 0 ? (long?)v : null);
                        }
                        return list;
                    }
                default:
                    return null;
            }
        }
    }
}
