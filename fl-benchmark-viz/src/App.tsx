import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { DataProvider } from './context/DataContext';
import { OverviewPage } from './pages/OverviewPage';
import { CurvesPage } from './pages/CurvesPage';
import { TradeoffPage } from './pages/TradeoffPage';
import { MatrixPage } from './pages/MatrixPage';
import { AttacksPage } from './pages/AttacksPage';
import { ExplorerPage } from './pages/ExplorerPage';

function App() {
  return (
    <DataProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<OverviewPage />} />
            <Route path="curves" element={<CurvesPage />} />
            <Route path="tradeoff" element={<TradeoffPage />} />
            <Route path="matrix" element={<MatrixPage />} />
            <Route path="attacks" element={<AttacksPage />} />
            <Route path="explorer" element={<ExplorerPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </DataProvider>
  );
}

export default App;
