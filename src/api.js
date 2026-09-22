import axios from "axios";

const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

export const api = axios.create({ baseURL: API_BASE });

export const figureUrl = (name) => `${API_BASE}/api/figure/${name}`;
