const nodeConsole = require('console');

export function createSimpleLogger(logTag: string, overrideConsole = false) {
    const theconsole = overrideConsole ? nodeConsole : console;
    return {
        log: (...args: any[]) => {
            theconsole.log(`[${logTag}]:`, ...args)
        },
        debug: (...args: any[]) => {
            theconsole.debug(`[${logTag}]:`, ...args)
        },
        info: (...args: any[]) => {
            theconsole.info(`[${logTag}]:`, ...args)
        },
        warn: (...args: any[]) => {
            theconsole.warn(`[${logTag}]:`, ...args)
        }
    };
}
