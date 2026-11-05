import { useCallback, useEffect, useState } from "react";
import { getErrorMessage, getEvaluationReport, listEvaluationReports } from "../services/api";
import type { EvaluationReport, EvaluationReportInfo } from "../types";

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export function useReportList(): AsyncState<EvaluationReportInfo[]> {
  const [data, setData] = useState<EvaluationReportInfo[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listEvaluationReports()
      .then((result) => !cancelled && (setData(result.reports), setError(null)))
      .catch((err) => !cancelled && setError(getErrorMessage(err)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);
  return { data, loading, error, reload };
}

export function useReport(reportId: string | null): AsyncState<EvaluationReport> {
  const [data, setData] = useState<EvaluationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!reportId) {
      setData(null);
      return;
    }
    // Ignore a slow response for a report the user has already switched away from.
    let cancelled = false;
    setLoading(true);
    getEvaluationReport(reportId)
      .then((result) => !cancelled && (setData(result), setError(null)))
      .catch((err) => !cancelled && setError(getErrorMessage(err)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [reportId, attempt]);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);
  return { data, loading, error, reload };
}
