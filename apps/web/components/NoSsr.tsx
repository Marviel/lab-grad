import dynamic from 'next/dynamic'
import React, { PropsWithChildren, useEffect, useState } from 'react'

/**
 * This component ensures that its children are only rendered on the client.
 */
const NoSsr = ({ children, ...delegated }: PropsWithChildren) => {
    const [hasMounted, setHasMounted] = useState(false);
    useEffect(() => {
        setHasMounted(true);
    }, []);

    if (!hasMounted) {
        return null;
    }

    return <div {...delegated}>{children}</div>;
}

// This should be sufficient to cause no-ssr by itself, but it doesn't always work as we'd like.
export default dynamic(() => Promise.resolve(NoSsr), {
    ssr: false
})