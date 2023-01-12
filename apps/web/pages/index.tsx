import React, { useEffect, useMemo } from "react";

import NoSsr from "../components/NoSsr";
import { Web } from "./Web";

// We disable server side rendering for the entire application.
export default function App() {
  return (
    <NoSsr><Web /></NoSsr>
  )
}
