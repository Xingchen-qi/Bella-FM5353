using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("Monte Carlo Option Pricing Simulator");

        Console.Write("Initial stock price: ");
        double S0 = double.Parse(Console.ReadLine());

        Console.Write("Strike price: ");
        double K = double.Parse(Console.ReadLine());

        Console.Write("Risk-free rate: ");
        double r = double.Parse(Console.ReadLine());

        Console.Write("Volatility: ");
        double sigma = double.Parse(Console.ReadLine());

        Console.Write("Time to maturity: ");
        double T = double.Parse(Console.ReadLine());

        Console.Write("Number of steps: ");
        int n = int.Parse(Console.ReadLine());

        Console.Write("Number of simulations: ");
        int m = int.Parse(Console.ReadLine());
        
        Console.Write("Do you want to use Antithetic Variance Reduction? ");
        bool useAntithetic=Console.ReadLine().ToLower() == "yes";

        Console.Write("Do you want to use Control Variate?:");
        bool useControlVariate = Console.ReadLine().ToLower() == "yes";

        double dt = T / n;
        double a1 = (r - 0.5 * sigma * sigma) * dt; 
        double a2 = sigma * Math.Sqrt(dt); 
        double stt = Math.Exp(r * dt); 
        Random rand = new Random();
        double sumPayoff = 0.0;

        for (int j = 0; j < m; j++)
        {
            double s = S0;
            double sAnti = S0;
            double cv = 0.0;
            double cvAnti = 0.0;

            for (int i = 1; i <= n; i++)
            {
                double tau = (i - 1) * dt;
                double delta = BlackScholesDelta(s, tau, K, T, sigma, r);
                double z = BoxMuller(rand);
                double st = s * Math.Exp(a1 + a2 * z);
                cv += delta * (st - s * stt);
                s = st;

                if (useAntithetic)
                {
                    double deltaAnti = BlackScholesDelta(sAnti, tau, K, T, sigma, r);
                    double stback = sAnti * Math.Exp(a1 - a2 * z);
                    cvAnti += deltaAnti * (stback - sAnti * stt);
                    sAnti = stback;
                }
            }
            double CT = Math.Max(s - K, 0.0);
            if (useControlVariate)
                CT -= cv;

            if (useAntithetic)
            {
                double CTanti = Math.Max(sAnti - K, 0.0);
                if (useControlVariate)
                    CTanti -= cvAnti;

                CT = 0.5 * (CT + CTanti);
            }

            sumPayoff += CT;
        }
        double callPrice = Math.Exp(-r * T) * (sumPayoff / m);
        Console.WriteLine($"Call Price: {callPrice:F4}");
    }

    static double BlackScholesDelta(double S, double t, double K, double T,
                                    double sigma, double r)
    {
        double tau = T - t;
        if (tau <= 0) return (S > K ? 1.0 : 0.0);

        double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * tau)
                    / (sigma * Math.Sqrt(tau));
        return CumDensity(d1);
    }

    static double BoxMuller(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
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




