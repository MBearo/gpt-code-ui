const resolvedWebAddress = import.meta.env.VITE_WEB_ADDRESS ? import.meta.env.VITE_WEB_ADDRESS : "";

console.log('resolvedWebAddress',resolvedWebAddress)
const Config = {
    WEB_ADDRESS: resolvedWebAddress,
    API_ADDRESS: resolvedWebAddress + "/api"
}

export default Config;