/// Copyright (c) Sarah Kaiser. All rights reserved.
/// Licensed under the MIT license.

////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////

namespace QuantumPerceptron {

    open Microsoft.Quantum.Primitive;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Extensions.Convert;
    //open Microsoft.Quantum.Arrays;
    
    ////////////////////////////////////////////////////////////////////////
    /// Encoding dataset into qubits
    /// 
    /// # Summary
    /// Prepares two qubits in a state which represents a point in the training dataset.
    /// # Inputs
    /// data :       The input vector (single floating-point number)
    /// label :      The input label (0 or 1)
    /// dataQubit :  The qubit to be prepared in a state which represents input vector
    /// labelQubit : The qubit to be prepared in a state which represents the label 
    ///              (|0⟩ or |1⟩ for labels 0 or 1, respectively)
    ////////////////////////////////////////////////////////////////////////
    operation EncodeDataInQubits(
        data : Double, 
        label : Int, 
        dataQubit : Qubit, 
        labelQubit : Qubit) : Unit {

        // Make sure both qubits start in |0⟩ state.
        Reset(dataQubit);
        Reset(labelQubit);
        
        // Encode the input vector in dataQubit state using Ry rotation gate.
        // Note that the rotation angle has to be exactly "data" to be consistent with the labels we generate.
        Ry(data, dataQubit);
        
        // Encode the label in labelQubit state: |0⟩ or |1⟩ for labels 0 or 1
        if (label == 1) {
            X(labelQubit);
        }
    }
    
    
    ////////////////////////////////////////////////////////////////////////
    /// Single-shot single-point classification circuit
    ///
    /// # Summary
    /// Classifies a data point encoded as a qubit and validates the result against the expected label.
    /// # Inputs
    /// alpha      : The model parameter used for classification
    /// dataQubit  : The qubit which represents the input state
    /// labelQubit : The qubit which represents the expected label of the input state
    /// # Result
    /// true if the data point has been classified correctly, 
    /// false if it has been misclassified.
    ////////////////////////////////////////////////////////////////////////
    operation Classify(
        alpha : Double, 
        dataQubit : Qubit, 
        labelQubit : Qubit) : Bool {
        
        // Rotate the state of the data qubit by -alpha;
        // this will get it close to the |0⟩ state if the data point belonged to class 0,
        // and to the |1⟩ state if the data point belonged to class 1
        Ry(-alpha, dataQubit);

        // Now let's check with the `labelQubit` to see if we got it right!
        return Validate(dataQubit, labelQubit);

    }

    ////////////////////////////////////////////////////////////////////////
    /// Validating a single classification
    ///
    /// # Summary
    /// Validates a classified datapoint against against the expected label.
    /// # Inputs
    /// dataQubit  : The qubit which represents the classified state
    /// labelQubit : The qubit which represents the expected label of the input state
    /// # Result
    /// true if the data point has been classified correctly, 
    /// false if it has been misclassified.
    ////////////////////////////////////////////////////////////////////////
    operation Validate( 
        dataQubit : Qubit, 
        labelQubit : Qubit) : Bool {

        // Apply CNOT with data qubit as control and label qubit as target 
        // to compute XOR of the expected label and the computed label on labelQubit
        CNOT(dataQubit, labelQubit);

        // Measure the label qubit in computational basis

        // If the measurement result is |0⟩, XOR of the expected label and the computed label is 0, 
        // which means that the labels are the same and classification was correct.
        // Return the classification success
        return M(labelQubit) == Zero;
    }

    
    ////////////////////////////////////////////////////////////////////////
    /// Classify and Validate an entire dataset
    ///
    /// # Summary
    /// Given the value of the model parameter (average rotation angle of the data points in class 0),
    /// # Inputs
    /// alpha      : The model parameter used for classification
    /// dataPoints : An array of training vectors (individual floating-point numbers)
    /// labels     : An array of training labels (0 or 1)
    /// nIterations   : The number of times each point is classified; larger values give higher accuracy but longer run time
    /// # Result
    /// The success rate of classification for the given model parameter.
    ////////////////////////////////////////////////////////////////////////
    operation EstimateQuantumClassifierSuccessRate(
        alpha : Double, 
        dataPoints : Double[], 
        labels : Int[],
        nIterations: Int) : Double {
        
        Message($"Estimating classifier success rate at {alpha}...");
        
        let N = Length(dataPoints);
        // 
        //let nIterations = 201;
        // Define a mutable variable to store the number of correctly classified points in the dataset
        mutable nCorrectPoints = 0;
        
        // Allocate two qubits to be used in the classification
        using ((dataQubit, labelQubit) = (Qubit(), Qubit())) {
            
            // Iterate over all points of the dataset
            for ((dataPoint, label) in Zip(dataPoints, labels)) {
                
                // Define a mutable variable to store the number of successful classification runs
                mutable nCorrectClassificationRuns = 0;
                
                // Classify i-th data point by running classification circuit nIterations times
                for (j in 1 .. nIterations) {
                    // Prepare data qubit and label qubit in a state which encodes the j-th data point
                    EncodeDataInQubits(dataPoint, label, dataQubit, labelQubit);

                    // Run classification on the prepared qubits and count the runs when it succeeded
                    if (Classify(alpha, dataQubit, labelQubit)) {
                        set nCorrectClassificationRuns = nCorrectClassificationRuns + 1;
                    }
                }
                
                // The point in the dataset has been classified correctly if 
                // the share of runs on which classification succeeded is greater than 50%.
                if (nCorrectClassificationRuns * 2 > nIterations) {
                    set nCorrectPoints = nCorrectPoints + 1;
                }
            }

            // Clean up both qubits before deallocating them using library operation Reset.
            ResetAll([dataQubit, labelQubit]);
        }
        
        // Return the success rate of the classification (the percentage of points that have been classified correctly)
        // Note that you need IntAsDouble library function to convert integer numbers to doubles explicitly
        return ToDouble(nCorrectPoints) / ToDouble(N);
    }
}
