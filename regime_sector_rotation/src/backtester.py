import os
import pandas as pd

def generate_quantstats_tearsheet(net_returns, benchmark_returns, output_filename="ML_Sector_Rotation_Tearsheet.html"):
    """
    Constructs an institutional-grade performance tearsheet using QuantStats.
    Output is exported as a standalone interactive HTML file.
    """
    print("Initializing Institutional Quant Performance Analysis...")
    
    # 1. Clean and align return series (QuantStats requires timezone-naive daily or periodic series)
    strat_series = pd.Series(net_returns, index=pd.to_datetime(net_returns.index))
    strat_series.name = 'ML Rotation Strategy (Net)'
    
    bench_series = pd.Series(benchmark_returns, index=pd.to_datetime(benchmark_returns.index))
    bench_series.name = 'Benchmark (11 Sector EW)'
    
    if strat_series.index.tz is not None:
        strat_series.index = strat_series.index.tz_localize(None)
    if bench_series.index.tz is not None:
        bench_series.index = bench_series.index.tz_localize(None)
        
    strat_series = strat_series.dropna()
    bench_series = bench_series.dropna()
    
    # Determine export path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', output_filename)
    output_path = os.path.abspath(output_path)
    
    # 2. Trigger QuantStats compilation
    try:
        import quantstats as qs
        print(f"Compiling QuantStats HTML tearsheet... Saving to: {output_path}")
        qs.reports.html(
            strat_series, 
            benchmark=bench_series,
            output=output_path,
            title='Regime-Conditioned Sector Rotation Strategy'
        )
        print("Tearsheet generation completed successfully.")
        return output_path
    except ImportError:
        print("Warning: 'quantstats' library is not installed. Skipping HTML tearsheet generation.")
        print("You can install it manually using: pip install quantstats")
        return None
    except Exception as e:
        print(f"Error compiling QuantStats tearsheet: {e}")
        return None
