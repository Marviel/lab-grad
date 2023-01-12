import React, { useEffect, useMemo } from "react";
import { Button } from "ui";
import { Value } from '@lab-grad/lib';
import { makeVar, ApolloProvider, ApolloClient, InMemoryCache } from '@apollo/client';
import dynamic from 'next/dynamic'
import ExCanvas from "../components/Excali/ExCanvas";
import { CytoCanvas } from "../components/Cyto/CytoCanvas";

export function Web() {
    const ac = useMemo(() => {
        return new ApolloClient({
            cache: new InMemoryCache({
            })
        })
    }, [])

    return (
        <ApolloProvider client={ac}>
            <div style={{ position: 'absolute', top: '0px', left: '0px', height: '100vw', width: '100vw' }}>
                <div style={{ height: '100vw', width: '100vw' }}>
                    <CytoCanvas />
                </div>
            </div>
        </ApolloProvider>
    );
}