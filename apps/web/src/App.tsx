import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/Layout'
import { I2VStudio } from './pages/I2VStudio'
import { ComingSoon } from './pages/ComingSoon'
import { Dashboard } from './pages/Dashboard'
import { TTSStudio } from './pages/TTSStudio'
import { ImageStudio } from './pages/ImageStudio'
import { ProjectList } from './pages/ProjectList'

function App() {
    return (
        <Layout>
            <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/tools/tts" element={<TTSStudio />} />
                <Route path="/tools/imagegen" element={<ImageStudio />} />
                <Route path="/tools/i2v" element={<I2VStudio />} />
                <Route path="/projects" element={<ProjectList />} />
                <Route path="/projects/new" element={<ComingSoon title="New Project" />} />
                <Route path="/projects/:id" element={<ComingSoon title="Project Detail" />} />
                <Route path="/assets" element={<ComingSoon title="Assets" />} />
            </Routes>
        </Layout>
    )
}

export default App
