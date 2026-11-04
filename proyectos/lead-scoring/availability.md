# Predictor availability audit

Project documentation supports the four form fields, independent capture logs remain pending.

| field | role | form_model | temporal_evidence | required_evidence |
| --- | --- | --- | --- | --- |
| id | identity_only | excluded | source_identifier | never_use_as_predictor |
| compra | target | excluded | outcome_timestamp_absent | define_outcome_window_and_maturity |
| no_enviar_email | contact_restriction | excluded | policy_field | preserve_as_auxiliary |
| no_llamar | contact_restriction | excluded | policy_field | preserve_as_auxiliary |
| origen | predictor | included | documented_form_field_not_independently_verified | source_capture_and_update_timestamps |
| fuente | predictor | included | documented_form_field_not_independently_verified | source_capture_and_update_timestamps |
| ambito | predictor | included | documented_form_field_not_independently_verified | source_capture_and_update_timestamps |
| ocupacion | predictor | included | documented_form_field_not_independently_verified | source_capture_and_update_timestamps |
| visitas_total | predictor | excluded | unverified_accumulation_window | immutable_snapshot_at_form_submission |
| tiempo_en_site_total | predictor | excluded | unverified_accumulation_window | immutable_snapshot_at_form_submission |
| paginas_vistas_visita | predictor | excluded | unverified_accumulation_window | immutable_snapshot_at_form_submission |
| ult_actividad | predictor | excluded | unverified_event_and_update_time | event_and_database_write_timestamps |
| ult_actividad=SMS Sent | category_not_original_column | excluded | unverified_event_and_update_time | SMS_event_and_database_write_timestamps_before_score |
| score_actividad | predictor | excluded | unverified_computation_time | formula_inputs_and_snapshot_before_score |
| score_perfil | predictor | excluded | unverified_computation_time | formula_inputs_and_snapshot_before_score |
| usuario_nuevo | derived_not_original_column | excluded | derived_from_missing_scores | same_timing_evidence_as_both_scores |
| descarga_lm | predictor | excluded | unverified_event_time | download_event_and_database_write_timestamps |
| conociste_google | predictor | excluded | unverified_capture_time | source_capture_and_update_timestamps |
| conociste_revista | predictor | excluded | unverified_capture_time | source_capture_and_update_timestamps |
| conociste_periodico | predictor | excluded | unverified_capture_time | source_capture_and_update_timestamps |
| conociste_youtube | predictor | excluded | unverified_capture_time | source_capture_and_update_timestamps |
| conociste_facebook | predictor | excluded | unverified_capture_time | source_capture_and_update_timestamps |
| conociste_referencias | predictor | excluded | unverified_capture_time | source_capture_and_update_timestamps |
