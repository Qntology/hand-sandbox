use crate::fusion::{Band, Predictor};
use crate::ttm::TtmForecaster;

pub struct TtmPredictor<'a>(pub &'a TtmForecaster);

impl Predictor for TtmPredictor<'_> {
    fn name(&self) -> &str {
        "ttm"
    }

    fn min_history(&self) -> usize {
        self.0.min_history()
    }

    fn max_horizon(&self) -> usize {
        self.0.prediction_length()
    }

    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let out = self.0.forecast(series)?;
        let h = horizon.min(out.point.len());
        let point: Vec<f64> = out.point.iter().take(h).copied().collect();
        let pick = |level: f64| -> Option<Vec<f64>> {
            let q = out.quantiles.as_ref()?;
            let i = out.quantile_levels.iter().position(|l| (l - level).abs() < 1e-6)?;
            Some(q.get(i)?.iter().take(h).copied().collect())
        };
        Ok(Band { lo: pick(0.1), hi: pick(0.9), point })
    }
}
