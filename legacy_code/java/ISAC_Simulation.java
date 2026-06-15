import java.util.Random;

public class ISAC_Simulation {
    // TRACK A: SEFDM Signal Generation
    public static Complex[] generateSEFDM(Complex[] symbols, double alpha, int N) {
        Complex[] x = new Complex[N];
        double scale = 1.0 / Math.sqrt(N);

        for (int n = 0; n < N; n++) {
            Complex sum = new Complex(0, 0);
            for (int k = 0; k < N; k++) {
                // Squeezed phase angle: 2 * pi * n * k * alpha / N
                double angle = (2 * Math.PI * n * k * alpha) / N;
                Complex expTerm = new Complex(Math.cos(angle), Math.sin(angle));
                // s_k * e^(j * angle)
                Complex term = symbols[k].multiply(expTerm);
                sum = sum.add(term);
            }
            x[n] = sum.scale(scale);
        }
        return x;
    }
    // TRACK B: IEG-IRS Channel Formulation
    public static Complex calculateIEG_IRS_Channel(Complex[] h_BI, Complex[] h_IU, Complex[] groupPhases, int[] elementToGroupMap) {
        Complex h_eff = new Complex(0, 0);
        int numElements = h_BI.length;

        for (int i = 0; i < numElements; i++) {
            // Find which group this element belongs to
            int groupId = elementToGroupMap[i];
            Complex sharedPhaseShift = groupPhases[groupId];
            // Cascaded link for this specific element: h_BI * phase * h_IU
            Complex cascadedElement = h_BI[i].multiply(sharedPhaseShift).multiply(h_IU[i]);
            // Add to the total effective channel
            h_eff = h_eff.add(cascadedElement);
        }
        return h_eff;
    }
    // TRACK C: The Synthesis (Combined Signal)
    public static Complex[] generateReceivedSignal(Complex[] txSignal, Complex effectiveChannel, double noiseVariance) {
        Complex[] rxSignal = new Complex[txSignal.length];
        Random rand = new Random();

        for (int i = 0; i < txSignal.length; i++) {
            // y = H * x
            Complex cleanSignal = txSignal[i].multiply(effectiveChannel);
            
            // Generate basic Additive White Gaussian Noise (AWGN)
            double noiseReal = rand.nextGaussian() * Math.sqrt(noiseVariance / 2);
            double noiseImag = rand.nextGaussian() * Math.sqrt(noiseVariance / 2);
            Complex noise = new Complex(noiseReal, noiseImag);

            // y = H * x + w
            rxSignal[i] = cleanSignal.add(noise);
        }
        return rxSignal;
    }
    // MAIN: Running the parallel simulation
    public static void main(String[] args) {
        int N = 64; // Number of subcarriers
        double alpha = 0.8; // SEFDM Compression factor (1.0 = standard OFDM)
        int numIRSElements = 1000;
        int numGroups = 4; // IEG-IRS groupings

        // 1. Setup mock data
        Complex[] dataSymbols = new Complex[N]; // The QAM/PSK symbols
        for(int i=0; i<N; i++) dataSymbols[i] = new Complex(1, 1); // Mock 1+1j symbol

        Complex[] h_BI = new Complex[numIRSElements]; // BS to IRS channel
        Complex[] h_IU = new Complex[numIRSElements]; // IRS to User channel
        int[] elementToGroupMap = new int[numIRSElements]; // Which element goes to which group
        Complex[] groupPhases = new Complex[numGroups]; // The optimized phase shifts for the 4 groups

        // Populate mock channel data and groupings
        for(int i=0; i<numIRSElements; i++) {
            h_BI[i] = new Complex(0.5, 0.1); 
            h_IU[i] = new Complex(0.4, -0.2);
            elementToGroupMap[i] = i % numGroups; // Distribute elements evenly across groups
        }
        for(int q=0; q<numGroups; q++) {
            groupPhases[q] = new Complex(Math.cos(Math.PI/4), Math.sin(Math.PI/4)); // 45 degree shift
        }

        // 2. Execute Track A (SEFDM)
        Complex[] transmittedSignal = generateSEFDM(dataSymbols, alpha, N);
        
        // 3. Execute Track B (IEG-IRS)
        Complex effectiveChannel = calculateIEG_IRS_Channel(h_BI, h_IU, groupPhases, elementToGroupMap);

        // 4. Execute Track C (Synthesis)
        double noisePower = 0.01;
        Complex[] receivedSignal = generateReceivedSignal(transmittedSignal, effectiveChannel, noisePower);

        System.out.println("Simulation Complete!");
        System.out.println("Effective IRS Channel: " + effectiveChannel.r + " + " + effectiveChannel.i + "j");
        System.out.println("Sample Received Signal [0]: " + receivedSignal[0].r + " + " + receivedSignal[0].i + "j");
    }
    // UTILITY: Lightweight Complex Number Class
    static class Complex {
        double r, i;
        public Complex(double real, double imag) {
            this.r = real;
            this.i = imag;
        }
        public Complex add(Complex b) {
            return new Complex(this.r + b.r, this.i + b.i);
        }
        public Complex multiply(Complex b) {
            return new Complex(this.r * b.r - this.i * b.i, this.r * b.i + this.i * b.r);
        }
        public Complex scale(double scalar) {
            return new Complex(this.r * scalar, this.i * scalar);
        }
    }
}
