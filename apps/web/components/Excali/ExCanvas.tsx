import React, { useEffect, useMemo, useState } from "react";
import { Button } from "ui";
import { Value } from '@lab-grad/lib';
import { makeVar, ApolloProvider, ApolloClient, InMemoryCache } from '@apollo/client';
// Cannot do it this way for ssr reasons?
// import Excalidraw from '@excalidraw/excalidraw';
const Excalidraw = dynamic(() => import('@excalidraw/excalidraw'), { ssr: false });
import dynamic from 'next/dynamic'
import initialData from "./initialData";

export default function ExCanvas() {
  const onChange = (elements: any, state: any) => {
    console.log("Elements :", elements, "State : ", state);
  };

  const onUsernameChange = (username: any) => {
    console.log("current username", username);
  };
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  const onResize = () => {
    setDimensions({
      width: window.innerWidth,
      height: window.innerHeight
    });
  };

  useEffect(() => {
    window.addEventListener("resize", onResize);

    return () => window.removeEventListener("resize", onResize);
  }, []);

  const { width, height } = dimensions;
  const options = { zenModeEnabled: true, viewBackgroundColor: "#AFEEEE" };
  return (
    <div style={{ height: '900px', width: '900px' }}>
      <Excalidraw
        // width={width}
        // height={height}
        initialData={initialData}
        onChange={onChange}
        user={{ name: "Excalidraw User" }}
        onUsernameChange={onUsernameChange}
        options={options}
        UIOptions={{ canvasActions: { loadScene: false } }}
      />
    </div>
  );
}