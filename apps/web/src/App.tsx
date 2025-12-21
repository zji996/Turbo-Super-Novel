import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/Layout'
import { I2VStudio } from './pages/I2VStudio'
import { ComingSoon } from './pages/ComingSoon'

function App() {
    return (
        <Layout>
            <Routes>
                <Route path="/" element={<Navigate to="/tools/i2v" replace />} />
                <Route path="/tools/i2v" element={<I2VStudio />} />
                <Route path="/projects" element={<ComingSoon title="Projects" />} />
                <Route path="/assets" element={<ComingSoon title="Assets" />} />
            </Routes>
        </Layout>
    )
}

export default App
