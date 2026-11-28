// Keep rounding at the point of display. Verification and plot coordinates use the original values.
const figures = (() => {
  const display = roundingMode => {
    const format = new Intl.NumberFormat("en-US", {
      maximumSignificantDigits: 4, useGrouping: false, roundingMode,
    }).format;
    return value => format(value === 0 ? 0 : value).replace("-", "−");
  };
  return {nearest: display("halfExpand"), upper: display("ceil"), lower: display("floor")};
})();

function signedFigure(value, format = figures.nearest) {
  return (value > 0 ? "+" : "") + format(value);
}
