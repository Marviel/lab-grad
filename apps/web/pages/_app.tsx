import type { AppProps } from "next/app";
import dynamic from "next/dynamic";
import React, { useState } from "react";
import NoSsr from "../components/NoSsr";


const App = ({ Component, pageProps }: AppProps) => {
    return <>
        <NoSsr>
            {/* <div>APP</div> */}
            <Component {...pageProps} />
        </NoSsr>
    </>;
};

export default dynamic(() => Promise.resolve(App), {
    ssr: false,
});