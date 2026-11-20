import React, {useState} from "react";
import {createRoot} from "react-dom/client";
import * as Tabs from "@radix-ui/react-tabs";
import * as ToggleGroup from "@radix-ui/react-toggle-group";

// Adapted from the shadcn/ui Toggle Group and Tabs listed on 21st.dev.
const ranges = {"1": "1Y", "3": "3Y", "5": "5Y", "0": "All"};
const rangeOrder = ["1", "3", "5", "0"];

function ChartRange({onChange}) {
  const [range, setRange] = useState("0");
  return (
    <ToggleGroup.Root type="single" className="ranges" aria-label="Chart period" value={range}
      onValueChange={(value) => {
        if (!value) return;
        setRange(value);
        onChange(Number(value));
      }}>
      {rangeOrder.map((value) => <ToggleGroup.Item key={value} value={value}>{ranges[value]}</ToggleGroup.Item>)}
    </ToggleGroup.Root>
  );
}

function FillTable({rows, empty}) {
  return (
    <div className="table-region fill-scroll" tabIndex={0} role="region" aria-label="Simulated fills">
      <table>
        <thead><tr>{["Date", "Ticker", "Side", "Shares", "Fill price"].map((name) => <th key={name} scope="col">{name}</th>)}</tr></thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={5}>{empty}</td></tr>}
          {rows.map(([date, ticker, side, units, price], index) => (
            <tr key={`${date}-${ticker}-${side}-${index}`}>
              <td>{date}</td><td>{ticker}</td><td className={side.toLowerCase()}>{side === "BUY" ? "Buy" : "Sell"}</td>
              <td>{units}</td><td>{price}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function mountChartRange(element, onChange) {
  createRoot(element).render(<ChartRange onChange={onChange} />);
}

export function mountActivity(element, allRows) {
  const root = createRoot(element);
  return (date, empty) => root.render(
    <Tabs.Root defaultValue="selected">
      <Tabs.List className="activity-tabs" aria-label="Fill history">
        <Tabs.Trigger value="selected">Selected date</Tabs.Trigger>
        <Tabs.Trigger value="all">All fills ({allRows.length})</Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="selected"><FillTable rows={allRows.filter((row) => row[0] === date)} empty={empty} /></Tabs.Content>
      <Tabs.Content value="all"><FillTable rows={allRows} empty="No archived fills." /></Tabs.Content>
    </Tabs.Root>
  );
}
