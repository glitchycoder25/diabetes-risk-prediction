import axios from "axios";

// Change this if you deploy the backend somewhere other than localhost.
const API_BASE = "http://localhost:8000";

export const api = axios.create({ baseURL: API_BASE });

export const figureUrl = (name) => `${API_BASE}/api/figure/${name}`;
