-- Скрипт структури сховища даних для MS SQL Server (схема «зірка»).
-- УВАГА: у цьому прототипі скрипт НЕ виконується і не тестувався на реальному сервері.
-- Застосунок відтворює ту саму структуру в SQLite (див. db.py).

CREATE TABLE dbo.Dim_Time (
    time_id      INT IDENTITY(1,1) PRIMARY KEY,
    [datetime]   DATETIME2(0) NOT NULL UNIQUE,
    [hour]       TINYINT      NOT NULL,
    day_of_week  TINYINT      NOT NULL,
    [month]      TINYINT      NOT NULL,
    is_weekend   BIT          NOT NULL,
    is_holiday   BIT          NOT NULL DEFAULT 0
);

CREATE TABLE dbo.Dim_Object (
    object_id    INT IDENTITY(1,1) PRIMARY KEY,
    name         NVARCHAR(200) NOT NULL UNIQUE,
    object_type  NVARCHAR(100) NULL,
    area_m2      DECIMAL(12,2) NULL,
    location     NVARCHAR(200) NULL
);

CREATE TABLE dbo.Dim_Model (
    model_id     INT IDENTITY(1,1) PRIMARY KEY,
    model_name   NVARCHAR(100) NOT NULL,
    mae          DECIMAL(14,4) NULL,
    rmse         DECIMAL(14,4) NULL,
    mape         DECIMAL(14,4) NULL,
    trained_at   DATETIME2(0)  NOT NULL
);

CREATE TABLE dbo.Fact_Consumption (
    consumption_id INT IDENTITY(1,1) PRIMARY KEY,
    time_id        INT NOT NULL REFERENCES dbo.Dim_Time(time_id),
    object_id      INT NOT NULL REFERENCES dbo.Dim_Object(object_id),
    energy_kwh     DECIMAL(14,3) NOT NULL,
    power_kw       DECIMAL(14,3) NULL,
    temperature_c  DECIMAL(6,2)  NULL,
    is_anomaly     BIT NOT NULL DEFAULT 0,
    CONSTRAINT UQ_Fact_Consumption UNIQUE (time_id, object_id)
);
CREATE INDEX ix_fact_consumption_time_obj ON dbo.Fact_Consumption(time_id, object_id);

CREATE TABLE dbo.Fact_Forecast (
    forecast_id   INT IDENTITY(1,1) PRIMARY KEY,
    time_id       INT NOT NULL REFERENCES dbo.Dim_Time(time_id),
    object_id     INT NOT NULL REFERENCES dbo.Dim_Object(object_id),
    model_id      INT NOT NULL REFERENCES dbo.Dim_Model(model_id),
    predicted_kwh DECIMAL(14,3) NOT NULL,
    horizon_h     INT NOT NULL
);
