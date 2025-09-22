using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("Monte Carlo Option Pricing Simulator");

        Console.Write("Initial stock price: ");
        double s0 = double.Parse(Console.ReadLine());

        Console.Write("Strike price: ");
        double k = double.Parse(Console.ReadLine());

        Console.Write("Risk-free rate: ");
        double r = double.Parse(Console.ReadLine());

        Console.Write("Volatility: ");
        double v = double.Parse(Console.ReadLine());

        Console.Write("Time to maturity: ");
        double t = double.Parse(Console.ReadLine());

        Console.Write("Number of steps: ");
        int steps = int.Parse(Console.ReadLine());

        Console.Write("Number of simulations: ");
        int sims = int.Parse(Console.ReadLine());
        
        Console.Write("Do you want to use Antithetic Variance Reduction? ");
        bool useAntithetic=Console.ReadLine().ToLower() == "yes";

        Console.Write("Do you want to use Control Variate?:");
        bool useControlVariate = Console.ReadLine().ToLower() == "yes";

        double[] call = new double[sims];
        double[] put= new double[sims];
        double[] stValues = new double[sims];
        double[] controlVars = new double[sims];
        Random rand = new Random();
        for (int i = 0; i < sims; i++)
        {
            double st = s0;
            double stAnti = s0;
            double dt = t / steps;

            for (int j = 0; j< steps; j++)
            {
                double z = BoxMuller(rand);
                double drift = (r-0.5*v*v)*dt;
                double vol  = v * Math.Sqrt(dt);

                st *= Math.Exp(drift + vol * z);
                if (useAntithetic)
                    stAnti *= Math.Exp(drift - vol * z);
            }
            double payoffCall = Math.Max(st - k, 0);
            double payoffPut = Math.Max(k - st, 0);

            if (useAntithetic)
            {
                payoffCall = (payoffCall + Math.Max(stAnti - k, 0)) / 2.0;
                payoffPut = (payoffPut + Math.Max(k - stAnti, 0)) / 2.0;
            }

            call[i] = payoffCall;
            put[i] = payoffPut;
            stValues[i] = st;
        }
        if (useControlVariate)
        {
            double d1 = (Math.Log(s0 / k) + (r + 0.5 * v * v) * t) / (v * Math.Sqrt(t));
            double delta = CumDensity(d1);
            double expectedST = s0 * Math.Exp(r * t);
            for (int i = 0; i < sims; i++)
            {
                controlVars[i] = delta * (stValues[i] - expectedST);
            }

            double meanCall = Avg(call);
            double meanControl = Avg(controlVars);

            double beta = Cov(call, controlVars, meanCall, meanControl) / Var(controlVars, meanControl);

            for (int i = 0; i < sims; i++)
            {
                call[i] = call[i] - beta * (controlVars[i] - 0.0);
            }
        }

        double call_avg = Avg(call);
        double put_avg = Avg(put);
        double call_se = StdErr(call, call_avg, sims);
        double put_se= StdErr(put, put_avg, sims);
        double call_price=Math.Exp(-r * t) * call_avg;
        double put_price= Math.Exp(-r * t) * put_avg;

        Console.WriteLine($"\nCall price: {call_price:F4} | StdErr: {call_se:F4}");
        Console.WriteLine($"Put  price: {put_price:F4} | StdErr: {put_se:F4}");
    }
    
    static double BoxMuller(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    static double Avg(double[] arr)
    {
        double sum = 0;
        foreach (var x in arr) sum += x;
        return sum / arr.Length;
    }

    static double StdErr(double[] arr, double mean, int n)
    {
        double sum = 0;
        foreach (var x in arr) sum += (x - mean) * (x - mean);
        double std = Math.Sqrt(sum / (n - 1));
        return std / Math.Sqrt(n);
    }

    static double Cov(double[] x, double[]y, double meanX, double meanY)
    {
        double sum = 0;
        for (int i = 0; i<x.Length; i++)
            sum += (x[i] - meanX) * (y[i] - meanY);
        return sum / (x.Length - 1);
    }
    static double Var(double[] x, double mean )
    {
        double sum = 0;
        for ( int i = 0; i < x.Length; i++)
            sum += (x[i] - mean) *(x[i] - mean);
            return sum / (x.Length - 1);
    }
    static double CumDensity(double z)
    {
        double p = 0.3275911;
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;

        int sign = z < 0.0 ? -1 : 1;
        double x = Math.Abs(z) / Math.Sqrt(2.0);
        double t = 1.0 / (1.0 + p * x);
        double erf = 1.0 - (((((a5 * t + a4) * t) + a3)
            * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * erf);
    }
}

