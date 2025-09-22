#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

// Forward declarations
class MarketData;
class RiskManager;
class Portfolio;
class Strategy;

// ============================================================================
// MARKET DATA STRUCTURES AND HANDLING
// ============================================================================

struct Tick {
    std::string symbol;
    double bid;
    double ask;
    double last;
    long long volume;
    std::chrono::system_clock::time_point timestamp;
    
    double midPrice() const { return (bid + ask) / 2.0; }
    double spread() const { return ask - bid; }
};

struct OHLCV {
    double open, high, low, close;
    long long volume;
    std::chrono::system_clock::time_point timestamp;
    
    double getReturn() const { return std::log(close / open); }
    double getTrueRange(const OHLCV& prev) const {
        return std::max({high - low, 
                        std::abs(high - prev.close), 
                        std::abs(low - prev.close)});
    }
};

class TimeSeries {
private:
    std::vector<OHLCV> data;
    std::string symbol;
    
public:
    TimeSeries(const std::string& sym) : symbol(sym) {}
    
    void addBar(const OHLCV& bar) {
        data.push_back(bar);
        // Keep only last 10000 bars for memory efficiency
        if (data.size() > 10000) {
            data.erase(data.begin());
        }
    }
    
    const std::vector<OHLCV>& getData() const { return data; }
    size_t size() const { return data.size(); }
    
    std::vector<double> getClosePrices(int lookback = -1) const {
        std::vector<double> prices;
        int start = (lookback > 0) ? std::max(0, (int)data.size() - lookback) : 0;
        
        for (size_t i = start; i < data.size(); ++i) {
            prices.push_back(data[i].close);
        }
        return prices;
    }
    
    std::vector<double> getReturns(int lookback = -1) const {
        std::vector<double> returns;
        auto prices = getClosePrices(lookback);
        
        for (size_t i = 1; i < prices.size(); ++i) {
            returns.push_back(std::log(prices[i] / prices[i-1]));
        }
        return returns;
    }
    
    double getVolatility(int lookback = 252) const {
        auto returns = getReturns(lookback);
        if (returns.size() < 2) return 0.0;
        
        double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0.0;
        
        for (double ret : returns) {
            variance += (ret - mean) * (ret - mean);
        }
        
        return std::sqrt(variance / (returns.size() - 1)) * std::sqrt(252.0);
    }
};

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

class TechnicalIndicators {
public:
    static std::vector<double> SMA(const std::vector<double>& prices, int period) {
        std::vector<double> sma;
        if (prices.size() < period) return sma;
        
        for (size_t i = period - 1; i < prices.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += prices[i - j];
            }
            sma.push_back(sum / period);
        }
        return sma;
    }
    
    static std::vector<double> EMA(const std::vector<double>& prices, int period) {
        std::vector<double> ema;
        if (prices.empty()) return ema;
        
        double alpha = 2.0 / (period + 1);
        ema.push_back(prices[0]);
        
        for (size_t i = 1; i < prices.size(); ++i) {
            ema.push_back(alpha * prices[i] + (1 - alpha) * ema.back());
        }
        return ema;
    }
    
    static std::vector<double> RSI(const std::vector<double>& prices, int period = 14) {
        std::vector<double> rsi;
        if (prices.size() < period + 1) return rsi;
        
        std::vector<double> gains, losses;
        for (size_t i = 1; i < prices.size(); ++i) {
            double change = prices[i] - prices[i-1];
            gains.push_back(std::max(0.0, change));
            losses.push_back(std::max(0.0, -change));
        }
        
        auto avgGains = EMA(gains, period);
        auto avgLosses = EMA(losses, period);
        
        for (size_t i = 0; i < avgGains.size(); ++i) {
            if (avgLosses[i] == 0) {
                rsi.push_back(100.0);
            } else {
                double rs = avgGains[i] / avgLosses[i];
                rsi.push_back(100.0 - (100.0 / (1.0 + rs)));
            }
        }
        return rsi;
    }
    
    static std::vector<double> BollingerBands(const std::vector<double>& prices, 
                                            int period = 20, double multiplier = 2.0) {
        auto sma = SMA(prices, period);
        std::vector<double> upperBand, lowerBand;
        
        for (size_t i = 0; i < sma.size(); ++i) {
            size_t startIdx = i + period - 1;
            double variance = 0.0;
            
            for (int j = 0; j < period; ++j) {
                double diff = prices[startIdx - j] - sma[i];
                variance += diff * diff;
            }
            
            double stdDev = std::sqrt(variance / period);
            upperBand.push_back(sma[i] + multiplier * stdDev);
            lowerBand.push_back(sma[i] - multiplier * stdDev);
        }
        
        // Return interleaved upper and lower bands
        std::vector<double> bands;
        for (size_t i = 0; i < upperBand.size(); ++i) {
            bands.push_back(upperBand[i]);
            bands.push_back(lowerBand[i]);
        }
        return bands;
    }
    
    static double ATR(const std::vector<OHLCV>& bars, int period = 14) {
        if (bars.size() < period + 1) return 0.0;
        
        std::vector<double> trueRanges;
        for (size_t i = 1; i < bars.size(); ++i) {
            trueRanges.push_back(bars[i].getTrueRange(bars[i-1]));
        }
        
        auto atr = EMA(trueRanges, period);
        return atr.empty() ? 0.0 : atr.back();
    }
};

// ============================================================================
// ADVANCED OPTION PRICING MODELS
// ============================================================================

class OptionPricingModels {
private:
    static double normalCDF(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
    
    static double normalPDF(double x) {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }

public:
    // Black-Scholes Model
    struct BSParams {
        double S, K, T, r, sigma, q = 0.0; // q = dividend yield
    };
    
    static double blackScholesCall(const BSParams& p) {
        if (p.T <= 0) return std::max(p.S - p.K, 0.0);
        
        double d1 = (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T) 
                   / (p.sigma * std::sqrt(p.T));
        double d2 = d1 - p.sigma * std::sqrt(p.T);
        
        return p.S * std::exp(-p.q * p.T) * normalCDF(d1) - 
               p.K * std::exp(-p.r * p.T) * normalCDF(d2);
    }
    
    static double blackScholesPut(const BSParams& p) {
        if (p.T <= 0) return std::max(p.K - p.S, 0.0);
        
        double d1 = (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T) 
                   / (p.sigma * std::sqrt(p.T));
        double d2 = d1 - p.sigma * std::sqrt(p.T);
        
        return p.K * std::exp(-p.r * p.T) * normalCDF(-d2) - 
               p.S * std::exp(-p.q * p.T) * normalCDF(-d1);
    }
    
    // Heston Model for stochastic volatility
    struct HestonParams {
        double S, K, T, r, q;
        double v0, kappa, theta, sigma, rho;
    };
    
    // Monte Carlo implementation for Heston model
    static double hestonMonteCarlo(const HestonParams& p, bool isCall = true, 
                                 int numSims = 100000, int numSteps = 252) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> norm(0.0, 1.0);
        
        double dt = p.T / numSteps;
        double sqrtDt = std::sqrt(dt);
        double payoffSum = 0.0;
        
        for (int sim = 0; sim < numSims; ++sim) {
            double S = p.S;
            double v = p.v0;
            
            for (int step = 0; step < numSteps; ++step) {
                double Z1 = norm(gen);
                double Z2 = p.rho * Z1 + std::sqrt(1 - p.rho * p.rho) * norm(gen);
                
                // Evolve variance using full truncation scheme
                double vNext = v + p.kappa * (p.theta - std::max(v, 0.0)) * dt + 
                              p.sigma * std::sqrt(std::max(v, 0.0)) * sqrtDt * Z2;
                v = std::max(vNext, 0.0);
                
                // Evolve stock price
                S *= std::exp((p.r - p.q - 0.5 * std::max(v, 0.0)) * dt + 
                             std::sqrt(std::max(v, 0.0)) * sqrtDt * Z1);
            }
            
            double payoff = isCall ? std::max(S - p.K, 0.0) : std::max(p.K - S, 0.0);
            payoffSum += payoff;
        }
        
        return std::exp(-p.r * p.T) * payoffSum / numSims;
    }
    
    // Binomial Tree Model
    static double binomialTree(double S, double K, double T, double r, double sigma,
                             bool isCall = true, bool isAmerican = false, int steps = 100) {
        double dt = T / steps;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp(r * dt) - d) / (u - d);
        double discount = std::exp(-r * dt);
        
        // Initialize option values at expiration
        std::vector<double> optionValues(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            double ST = S * std::pow(u, 2 * i - steps);
            optionValues[i] = isCall ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
        }
        
        // Backward induction
        for (int step = steps - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                double continuationValue = discount * (p * optionValues[i + 1] + 
                                                     (1 - p) * optionValues[i]);
                
                if (isAmerican) {
                    double ST = S * std::pow(u, 2 * i - step);
                    double exerciseValue = isCall ? std::max(ST - K, 0.0) : 
                                                   std::max(K - ST, 0.0);
                    optionValues[i] = std::max(continuationValue, exerciseValue);
                } else {
                    optionValues[i] = continuationValue;
                }
            }
        }
        
        return optionValues[0];
    }
};

// ============================================================================
// POSITION AND TRADE MANAGEMENT
// ============================================================================

enum class OrderType { MARKET, LIMIT, STOP, STOP_LIMIT };
enum class OrderSide { BUY, SELL };
enum class OrderStatus { PENDING, FILLED, CANCELLED, REJECTED };

struct Order {
    std::string orderId;
    std::string symbol;
    OrderType type;
    OrderSide side;
    double quantity;
    double price;
    double stopPrice;
    OrderStatus status;
    std::chrono::system_clock::time_point timestamp;
    double filledQuantity = 0.0;
    double avgFillPrice = 0.0;
    
    bool isComplete() const { return filledQuantity >= quantity; }
};

struct Position {
    std::string symbol;
    double quantity;
    double avgCost;
    double unrealizedPnL;
    double realizedPnL;
    std::chrono::system_clock::time_point lastUpdate;
    
    double getMarketValue(double currentPrice) const {
        return quantity * currentPrice;
    }
    
    double getTotalPnL(double currentPrice) const {
        return (currentPrice - avgCost) * quantity + realizedPnL;
    }
};

class Portfolio {
private:
    std::map<std::string, Position> positions;
    std::vector<Order> orders;
    double cash;
    double totalValue;
    std::map<std::string, double> currentPrices;
    
public:
    Portfolio(double initialCash = 1000000.0) : cash(initialCash), totalValue(initialCash) {}
    
    void updatePrice(const std::string& symbol, double price) {
        currentPrices[symbol] = price;
        updatePortfolioValue();
    }
    
    void addTrade(const std::string& symbol, double quantity, double price) {
        auto it = positions.find(symbol);
        
        if (it == positions.end()) {
            // New position
            positions[symbol] = {symbol, quantity, price, 0.0, 0.0, 
                               std::chrono::system_clock::now()};
        } else {
            // Update existing position
            Position& pos = it->second;
            
            if ((pos.quantity > 0 && quantity > 0) || (pos.quantity < 0 && quantity < 0)) {
                // Adding to position
                double newQuantity = pos.quantity + quantity;
                pos.avgCost = (pos.avgCost * pos.quantity + price * quantity) / newQuantity;
                pos.quantity = newQuantity;
            } else {
                // Reducing or closing position
                double closedQuantity = std::min(std::abs(quantity), std::abs(pos.quantity));
                pos.realizedPnL += closedQuantity * (price - pos.avgCost) * 
                                  (pos.quantity > 0 ? 1 : -1);
                pos.quantity += quantity;
                
                if (std::abs(pos.quantity) < 1e-8) {
                    positions.erase(it);
                }
            }
        }
        
        cash -= quantity * price;
        updatePortfolioValue();
    }
    
    void updatePortfolioValue() {
        totalValue = cash;
        for (auto& [symbol, pos] : positions) {
            auto priceIt = currentPrices.find(symbol);
            if (priceIt != currentPrices.end()) {
                totalValue += pos.getMarketValue(priceIt->second);
            }
        }
    }
    
    double getTotalValue() const { return totalValue; }
    double getCash() const { return cash; }
    
    const std::map<std::string, Position>& getPositions() const { return positions; }
    
    // Risk metrics
    double getPortfolioBeta(const std::map<std::string, double>& betas) const {
        double totalBeta = 0.0;
        double totalValue = getTotalValue();
        
        for (const auto& [symbol, pos] : positions) {
            auto priceIt = currentPrices.find(symbol);
            auto betaIt = betas.find(symbol);
            
            if (priceIt != currentPrices.end() && betaIt != betas.end()) {
                double weight = pos.getMarketValue(priceIt->second) / totalValue;
                totalBeta += weight * betaIt->second;
            }
        }
        return totalBeta;
    }
    
    double getConcentrationRisk() const {
        double maxPositionValue = 0.0;
        for (const auto& [symbol, pos] : positions) {
            auto priceIt = currentPrices.find(symbol);
            if (priceIt != currentPrices.end()) {
                maxPositionValue = std::max(maxPositionValue, 
                                          std::abs(pos.getMarketValue(priceIt->second)));
            }
        }
        return maxPositionValue / totalValue;
    }
};

// ============================================================================
// RISK MANAGEMENT SYSTEM
// ============================================================================

class RiskManager {
private:
    double maxPositionSize;
    double maxPortfolioVar;
    double maxDrawdown;
    double currentDrawdown;
    double peakValue;
    
    std::map<std::string, double> correlationMatrix;
    std::map<std::string, double> volatilities;
    
public:
    RiskManager(double maxPos = 0.1, double maxVar = 0.02, double maxDD = 0.2) 
        : maxPositionSize(maxPos), maxPortfolioVar(maxVar), maxDrawdown(maxDD),
          currentDrawdown(0.0), peakValue(0.0) {}
    
    bool checkPositionLimit(const std::string& symbol, double quantity, 
                          double price, const Portfolio& portfolio) {
        double positionValue = std::abs(quantity * price);
        double portfolioValue = portfolio.getTotalValue();
        
        return (positionValue / portfolioValue) <= maxPositionSize;
    }
    
    double calculateVaR(const Portfolio& portfolio, double confidence = 0.05,
                       int holdingPeriod = 1) {
        // Simplified VaR calculation using variance-covariance method
        double portfolioVariance = 0.0;
        auto positions = portfolio.getPositions();
        
        for (const auto& [symbol1, pos1] : positions) {
            auto vol1 = volatilities.find(symbol1);
            if (vol1 == volatilities.end()) continue;
            
            double weight1 = pos1.quantity / portfolio.getTotalValue();
            
            for (const auto& [symbol2, pos2] : positions) {
                auto vol2 = volatilities.find(symbol2);
                if (vol2 == volatilities.end()) continue;
                
                double weight2 = pos2.quantity / portfolio.getTotalValue();
                double correlation = getCorrelation(symbol1, symbol2);
                
                portfolioVariance += weight1 * weight2 * vol1->second * vol2->second * correlation;
            }
        }
        
        double portfolioVol = std::sqrt(portfolioVariance * holdingPeriod);
        
        // Normal distribution quantile for confidence level
        double zScore = (confidence == 0.05) ? -1.645 : -2.326; // 95% or 99% confidence
        
        return portfolio.getTotalValue() * zScore * portfolioVol;
    }
    
    void updateVolatility(const std::string& symbol, double vol) {
        volatilities[symbol] = vol;
    }
    
    void updateCorrelation(const std::string& symbol1, const std::string& symbol2, double corr) {
        correlationMatrix[symbol1 + "_" + symbol2] = corr;
        correlationMatrix[symbol2 + "_" + symbol1] = corr;
    }
    
    double getCorrelation(const std::string& symbol1, const std::string& symbol2) {
        if (symbol1 == symbol2) return 1.0;
        
        auto it = correlationMatrix.find(symbol1 + "_" + symbol2);
        return (it != correlationMatrix.end()) ? it->second : 0.0;
    }
    
    void updateDrawdown(double currentValue) {
        if (currentValue > peakValue) {
            peakValue = currentValue;
            currentDrawdown = 0.0;
        } else {
            currentDrawdown = (peakValue - currentValue) / peakValue;
        }
    }
    
    bool isDrawdownAcceptable() const {
        return currentDrawdown <= maxDrawdown;
    }
    
    // Stress testing
    struct StressScenario {
        std::string name;
        std::map<std::string, double> shocks; // symbol -> shock percentage
    };
    
    double runStressTest(const Portfolio& portfolio, const StressScenario& scenario) {
        double stressValue = portfolio.getCash();
        
        for (const auto& [symbol, pos] : portfolio.getPositions()) {
            double currentPrice = 100.0; // Would get from market data
            auto shockIt = scenario.shocks.find(symbol);
            
            if (shockIt != scenario.shocks.end()) {
                double shockedPrice = currentPrice * (1 + shockIt->second);
                stressValue += pos.quantity * shockedPrice;
            } else {
                stressValue += pos.quantity * currentPrice;
            }
        }
        
        return (stressValue - portfolio.getTotalValue()) / portfolio.getTotalValue();
    }
};

// ============================================================================
// TRADING STRATEGIES
// ============================================================================

class Strategy {
protected:
    std::string name;
    Portfolio* portfolio;
    RiskManager* riskManager;
    std::map<std::string, TimeSeries*> marketData;
    
public:
    Strategy(const std::string& stratName, Portfolio* port, RiskManager* risk)
        : name(stratName), portfolio(port), riskManager(risk) {}
    
    virtual ~Strategy() = default;
    virtual void onTick(const Tick& tick) = 0;
    virtual void onBar(const std::string& symbol, const OHLCV& bar) = 0;
    virtual std::vector<Order> generateSignals() = 0;
    
    void addMarketData(const std::string& symbol, TimeSeries* ts) {
        marketData[symbol] = ts;
    }
    
    const std::string& getName() const { return name; }
};

// Mean Reversion Strategy
class MeanReversionStrategy : public Strategy {
private:
    int lookbackPeriod;
    double entryThreshold;
    double exitThreshold;
    std::map<std::string, double> entryPrices;
    std::map<std::string, bool> inPosition;
    
public:
    MeanReversionStrategy(Portfolio* port, RiskManager* risk, 
                         int lookback = 20, double entryThresh = 2.0, double exitThresh = 0.5)
        : Strategy("MeanReversion", port, risk), lookbackPeriod(lookback),
          entryThreshold(entryThresh), exitThreshold(exitThresh) {}
    
    void onTick(const Tick& tick) override {
        // Process real-time tick data if needed
    }
    
    void onBar(const std::string& symbol, const OHLCV& bar) override {
        auto it = marketData.find(symbol);
        if (it == marketData.end()) return;
        
        it->second->addBar(bar);
        
        if (it->second->size() < lookbackPeriod + 1) return;
        
        auto prices = it->second->getClosePrices(lookbackPeriod);
        auto sma = TechnicalIndicators::SMA(prices, lookbackPeriod);
        
        if (sma.empty()) return;
        
        double currentPrice = bar.close;
        double meanPrice = sma.back();
        double stdDev = calculateStdDev(prices);
        
        double zScore = (currentPrice - meanPrice) / stdDev;
        
        bool currentlyInPosition = inPosition[symbol];
        
        if (!currentlyInPosition && std::abs(zScore) > entryThreshold) {
            // Enter position - buy if oversold, sell if overbought
            double positionSize = 1000.0; // Fixed size for simplicity
            if (zScore < -entryThreshold) {
                // Oversold - buy
                portfolio->addTrade(symbol, positionSize / currentPrice, currentPrice);
                entryPrices[symbol] = currentPrice;
                inPosition[symbol] = true;
            } else if (zScore > entryThreshold) {
                // Overbought - sell short
                portfolio->addTrade(symbol, -positionSize / currentPrice, currentPrice);
                entryPrices[symbol] = currentPrice;
                inPosition[symbol] = true;
            }
        } else if (currentlyInPosition && std::abs(zScore) < exitThreshold) {
            // Exit position
            auto positions = portfolio->getPositions();
            auto posIt = positions.find(symbol);
            if (posIt != positions.end()) {
                portfolio->addTrade(symbol, -posIt->second.quantity, currentPrice);
                inPosition[symbol] = false;
            }
        }
    }
    
    std::vector<Order> generateSignals() override {
        return {}; // Simplified - actual implementation would generate formal orders
    }
    
private:
    double calculateStdDev(const std::vector<double>& prices) {
        if (prices.size() < 2) return 1.0;
        
        double mean = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        
        for (double price : prices) {
            variance += (price - mean) * (price - mean);
        }
        
        return std::sqrt(variance / (prices.size() - 1));
    }
};

// Momentum Strategy
class MomentumStrategy : public Strategy {
private:
    int shortPeriod;
    int longPeriod;
    std::map<std::string, bool> inPosition;
    
public:
    MomentumStrategy(Portfolio* port, RiskManager* risk, int shortP = 12, int longP = 26)
        : Strategy("Momentum", port, risk), shortPeriod(shortP), longPeriod(longP) {}
    
    void onTick(const Tick& tick) override {}
    
    void onBar(const std::string& symbol, const OHLCV& bar) override {
        auto it = marketData.find(symbol);
        if (it == marketData.end()) return;
        
        it->second->addBar(bar);
        
        if (it->second->size() < longPeriod + 1) return;
        
        auto prices = it->second->getClosePrices();
        auto emaShort = TechnicalIndicators::EMA(prices, shortPeriod);
        auto emaLong = TechnicalIndicators::EMA(prices, longPeriod);
        
        if (emaShort.size() < 2 || emaLong.size() < 2) return;
        
        double currentShort = emaShort.back();
        double currentLong = emaLong.back();
        double prevShort = emaShort[emaShort.size() - 2];
        double prevLong = emaLong[emaLong.size() - 2];
        
        bool bullishCrossover = (prevShort <= prevLong) && (currentShort > currentLong);
        bool bearishCrossover = (prevShort >= prevLong) && (currentShort < currentLong);
        
        if (bullishCrossover && !inPosition[symbol]) {
            double positionSize = 1000.0;
            portfolio->addTrade(symbol, positionSize / bar.close, bar.close);
            inPosition[symbol] = true;
        } else if (bearishCrossover && inPosition[symbol]) {
            auto positions = portfolio->getPositions();
            auto posIt = positions.find(symbol);
            if (posIt != positions.end()) {
                portfolio->addTrade(symbol, -posIt->second.quantity, bar.close);
                inPosition[symbol] = false;
            }
        }
    }
    
    std::vector<Order> generateSignals() override {
        return {};
    }
};

// ============================================================================
// BACKTESTING ENGINE
// ============================================================================

class BacktestEngine {
private:
    std::vector<Strategy*> strategies;
    std::map<std::string, TimeSeries> historicalData;
    Portfolio portfolio;
    RiskManager riskManager;
    
    struct PerformanceMetrics {
        double totalReturn;
        double annualizedReturn;
        double volatility;
        double sharpeRatio;
        double maxDrawdown;
        double calmarRatio;
        int numTrades;
        double winRate;
        double avgWin;
        double avgLoss;
        double profitFactor;
        double beta;
        double alpha;
        double informationRatio;
        double sortinoRatio;
    };
    
    std::vector<double> portfolioValues;
    std::vector<double> drawdowns;
    std::vector<std::chrono::system_clock::time_point> dates;
    
public:
    BacktestEngine(double initialCapital = 1000000.0) 
        : portfolio(initialCapital), riskManager() {}
    
    void addStrategy(Strategy* strategy) {
        strategies.push_back(strategy);
        strategy->addMarketData("AAPL", &historicalData["AAPL"]);
        strategy->addMarketData("GOOGL", &historicalData["GOOGL"]);
        strategy->addMarketData("MSFT", &historicalData["MSFT"]);
    }
    
    void loadHistoricalData(const std::string& symbol, const std::vector<OHLCV>& data) {
        for (const auto& bar : data) {
            historicalData[symbol].addBar(bar);
        }
    }
    
    PerformanceMetrics runBacktest(std::chrono::system_clock::time_point startDate,
                                 std::chrono::system_clock::time_point endDate) {
        portfolioValues.clear();
        drawdowns.clear();
        dates.clear();
        
        // Simulate trading day by day
        for (auto& [symbol, timeSeries] : historicalData) {
            const auto& data = timeSeries.getData();
            
            for (const auto& bar : data) {
                if (bar.timestamp >= startDate && bar.timestamp <= endDate) {
                    // Update portfolio with current prices
                    portfolio.updatePrice(symbol, bar.close);
                    
                    // Run strategies
                    for (auto* strategy : strategies) {
                        strategy->onBar(symbol, bar);
                    }
                    
                    // Record portfolio metrics
                    portfolioValues.push_back(portfolio.getTotalValue());
                    dates.push_back(bar.timestamp);
                    
                    // Update risk metrics
                    riskManager.updateDrawdown(portfolio.getTotalValue());
                    riskManager.updateVolatility(symbol, timeSeries.getVolatility());
                }
            }
        }
        
        return calculateMetrics();
    }
    
private:
    PerformanceMetrics calculateMetrics() {
        PerformanceMetrics metrics = {};
        
        if (portfolioValues.size() < 2) return metrics;
        
        // Calculate returns
        std::vector<double> returns;
        for (size_t i = 1; i < portfolioValues.size(); ++i) {
            returns.push_back((portfolioValues[i] - portfolioValues[i-1]) / portfolioValues[i-1]);
        }
        
        // Total and annualized return
        metrics.totalReturn = (portfolioValues.back() - portfolioValues.front()) / portfolioValues.front();
        
        double daysTradedYears = returns.size() / 252.0;
        metrics.annualizedReturn = std::pow(1 + metrics.totalReturn, 1.0 / daysTradedYears) - 1;
        
        // Volatility
        double meanReturn = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - meanReturn) * (ret - meanReturn);
        }
        metrics.volatility = std::sqrt(variance / (returns.size() - 1)) * std::sqrt(252);
        
        // Sharpe Ratio (assuming risk-free rate of 2%)
        double riskFreeRate = 0.02;
        metrics.sharpeRatio = (metrics.annualizedReturn - riskFreeRate) / metrics.volatility;
        
        // Maximum Drawdown
        double peak = portfolioValues[0];
        metrics.maxDrawdown = 0.0;
        for (double value : portfolioValues) {
            if (value > peak) peak = value;
            double drawdown = (peak - value) / peak;
            metrics.maxDrawdown = std::max(metrics.maxDrawdown, drawdown);
        }
        
        // Calmar Ratio
        metrics.calmarRatio = metrics.maxDrawdown > 0 ? 
                             metrics.annualizedReturn / metrics.maxDrawdown : 0.0;
        
        // Sortino Ratio (downside deviation)
        double downsideVariance = 0.0;
        int downsideDays = 0;
        for (double ret : returns) {
            if (ret < 0) {
                downsideVariance += ret * ret;
                downsideDays++;
            }
        }
        
        if (downsideDays > 0) {
            double downsideDeviation = std::sqrt(downsideVariance / downsideDays) * std::sqrt(252);
            metrics.sortinoRatio = (metrics.annualizedReturn - riskFreeRate) / downsideDeviation;
        }
        
        // Trade analysis would require tracking individual trades
        metrics.numTrades = 0; // Simplified
        metrics.winRate = 0.0;
        metrics.avgWin = 0.0;
        metrics.avgLoss = 0.0;
        metrics.profitFactor = 0.0;
        
        return metrics;
    }
    
public:
    void printResults(const PerformanceMetrics& metrics) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=============== BACKTEST RESULTS ===============" << std::endl;
        std::cout << "Total Return:           " << metrics.totalReturn * 100 << "%" << std::endl;
        std::cout << "Annualized Return:      " << metrics.annualizedReturn * 100 << "%" << std::endl;
        std::cout << "Volatility:             " << metrics.volatility * 100 << "%" << std::endl;
        std::cout << "Sharpe Ratio:           " << metrics.sharpeRatio << std::endl;
        std::cout << "Sortino Ratio:          " << metrics.sortinoRatio << std::endl;
        std::cout << "Maximum Drawdown:       " << metrics.maxDrawdown * 100 << "%" << std::endl;
        std::cout << "Calmar Ratio:           " << metrics.calmarRatio << std::endl;
        std::cout << "Number of Trades:       " << metrics.numTrades << std::endl;
        std::cout << "Final Portfolio Value:  $" << portfolioValues.back() << std::endl;
        std::cout << "===============================================" << std::endl;
    }
    
    // Monte Carlo simulation for strategy robustness
    std::vector<PerformanceMetrics> monteCarloAnalysis(int numSimulations = 1000) {
        std::vector<PerformanceMetrics> results;
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int sim = 0; sim < numSimulations; ++sim) {
            // Bootstrap returns
            std::vector<double> returns;
            for (size_t i = 1; i < portfolioValues.size(); ++i) {
                returns.push_back((portfolioValues[i] - portfolioValues[i-1]) / portfolioValues[i-1]);
            }
            
            std::uniform_int_distribution<> dist(0, returns.size() - 1);
            std::vector<double> bootstrappedReturns;
            
            for (size_t i = 0; i < returns.size(); ++i) {
                bootstrappedReturns.push_back(returns[dist(gen)]);
            }
            
            // Create new portfolio value series
            std::vector<double> simPortfolioValues;
            simPortfolioValues.push_back(portfolioValues[0]);
            
            for (double ret : bootstrappedReturns) {
                double newValue = simPortfolioValues.back() * (1 + ret);
                simPortfolioValues.push_back(newValue);
            }
            
            // Calculate metrics for this simulation
            std::vector<double> originalValues = portfolioValues;
            portfolioValues = simPortfolioValues;
            PerformanceMetrics simMetrics = calculateMetrics();
            portfolioValues = originalValues;
            
            results.push_back(simMetrics);
        }
        
        return results;
    }
};

// ============================================================================
// PORTFOLIO OPTIMIZATION
// ============================================================================

class PortfolioOptimizer {
private:
    std::vector<std::string> symbols;
    std::vector<std::vector<double>> returns;
    std::vector<std::vector<double>> covarianceMatrix;
    std::vector<double> expectedReturns;
    
public:
    void addAsset(const std::string& symbol, const std::vector<double>& assetReturns) {
        symbols.push_back(symbol);
        returns.push_back(assetReturns);
    }
    
    void calculateStatistics() {
        int n = symbols.size();
        expectedReturns.resize(n);
        covarianceMatrix.resize(n, std::vector<double>(n));
        
        // Calculate expected returns
        for (int i = 0; i < n; ++i) {
            expectedReturns[i] = std::accumulate(returns[i].begin(), returns[i].end(), 0.0) 
                               / returns[i].size();
        }
        
        // Calculate covariance matrix
        int T = returns[0].size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double covariance = 0.0;
                for (int t = 0; t < T; ++t) {
                    covariance += (returns[i][t] - expectedReturns[i]) * 
                                 (returns[j][t] - expectedReturns[j]);
                }
                covarianceMatrix[i][j] = covariance / (T - 1);
            }
        }
    }
    
    // Mean-Variance Optimization (simplified)
    std::vector<double> optimizePortfolio(double targetReturn) {
        int n = symbols.size();
        std::vector<double> weights(n, 1.0 / n); // Equal weights as starting point
        
        // This is a simplified optimization - in practice would use quadratic programming
        // For demonstration, we'll use a simple heuristic approach
        
        double learningRate = 0.01;
        int iterations = 1000;
        
        for (int iter = 0; iter < iterations; ++iter) {
            // Calculate current portfolio return and risk
            double portfolioReturn = 0.0;
            double portfolioVariance = 0.0;
            
            for (int i = 0; i < n; ++i) {
                portfolioReturn += weights[i] * expectedReturns[i];
                for (int j = 0; j < n; ++j) {
                    portfolioVariance += weights[i] * weights[j] * covarianceMatrix[i][j];
                }
            }
            
            // Update weights to move towards target return while minimizing risk
            std::vector<double> newWeights(n);
            double sumWeights = 0.0;
            
            for (int i = 0; i < n; ++i) {
                // Gradient-based update (simplified)
                double returnGradient = expectedReturns[i] - portfolioReturn;
                double riskGradient = 0.0;
                
                for (int j = 0; j < n; ++j) {
                    riskGradient += 2 * weights[j] * covarianceMatrix[i][j];
                }
                
                newWeights[i] = weights[i] + learningRate * 
                               (returnGradient * (targetReturn - portfolioReturn) - riskGradient);
                newWeights[i] = std::max(0.0, newWeights[i]); // No short selling
                sumWeights += newWeights[i];
            }
            
            // Normalize weights
            if (sumWeights > 0) {
                for (int i = 0; i < n; ++i) {
                    weights[i] = newWeights[i] / sumWeights;
                }
            }
        }
        
        return weights;
    }
    
    // Black-Litterman model implementation (simplified)
    std::vector<double> blackLittermanOptimization(const std::vector<double>& marketWeights,
                                                  const std::vector<double>& views,
                                                  const std::vector<double>& viewConfidence,
                                                  double tau = 0.025) {
        int n = symbols.size();
        
        // Implied equilibrium returns
        std::vector<double> impliedReturns(n);
        double marketReturn = 0.0;
        double marketVariance = 0.0;
        
        for (int i = 0; i < n; ++i) {
            marketReturn += marketWeights[i] * expectedReturns[i];
            for (int j = 0; j < n; ++j) {
                marketVariance += marketWeights[i] * marketWeights[j] * covarianceMatrix[i][j];
            }
        }
        
        double riskAversion = marketReturn / marketVariance;
        
        for (int i = 0; i < n; ++i) {
            impliedReturns[i] = riskAversion * marketWeights[i] * marketVariance;
        }
        
        // Combine with investor views (simplified implementation)
        std::vector<double> newExpectedReturns(n);
        for (int i = 0; i < n; ++i) {
            if (i < views.size() && viewConfidence[i] > 0) {
                double confidence = viewConfidence[i];
                newExpectedReturns[i] = (impliedReturns[i] + confidence * views[i]) / 
                                       (1 + confidence);
            } else {
                newExpectedReturns[i] = impliedReturns[i];
            }
        }
        
        // Use updated returns for optimization
        expectedReturns = newExpectedReturns;
        return optimizePortfolio(marketReturn);
    }
    
    void printOptimizationResults(const std::vector<double>& weights) {
        std::cout << "\n============= PORTFOLIO OPTIMIZATION RESULTS =============" << std::endl;
        
        double portfolioReturn = 0.0;
        double portfolioVariance = 0.0;
        
        for (size_t i = 0; i < symbols.size(); ++i) {
            std::cout << symbols[i] << ": " << std::fixed << std::setprecision(2) 
                     << weights[i] * 100 << "%" << std::endl;
            
            portfolioReturn += weights[i] * expectedReturns[i];
            for (size_t j = 0; j < symbols.size(); ++j) {
                portfolioVariance += weights[i] * weights[j] * covarianceMatrix[i][j];
            }
        }
        
        double portfolioVol = std::sqrt(portfolioVariance);
        double sharpeRatio = portfolioReturn / portfolioVol;
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\nExpected Return: " << portfolioReturn * 100 << "%" << std::endl;
        std::cout << "Expected Volatility: " << portfolioVol * 100 << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << sharpeRatio << std::endl;
        std::cout << "=========================================================" << std::endl;
    }
};

// ============================================================================
// MAIN DEMONSTRATION
// ============================================================================

// Generate sample market data
std::vector<OHLCV> generateSampleData(const std::string& symbol, int days = 1000) {
    std::vector<OHLCV> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normalDist(0.0, 0.02);
    
    double price = 100.0;
    auto startTime = std::chrono::system_clock::now() - std::chrono::hours(24 * days);
    
    for (int i = 0; i < days; ++i) {
        double dailyReturn = normalDist(gen);
        
        // Add some trend based on symbol
        if (symbol == "AAPL") dailyReturn += 0.0003;
        if (symbol == "GOOGL") dailyReturn += 0.0002;
        if (symbol == "MSFT") dailyReturn += 0.0001;
        
        double newPrice = price * std::exp(dailyReturn);
        
        OHLCV bar;
        bar.open = price;
        bar.high = price * (1 + std::abs(normalDist(gen)) * 0.5);
        bar.low = price * (1 - std::abs(normalDist(gen)) * 0.5);
        bar.close = newPrice;
        bar.volume = 1000000 + static_cast<long long>(normalDist(gen) * 200000);
        bar.timestamp = startTime + std::chrono::hours(24 * i);
        
        data.push_back(bar);
        price = newPrice;
    }
    
    return data;
}

int main() {
    std::cout << "=== PROFESSIONAL QUANTITATIVE TRADING SYSTEM ===" << std::endl;
    
    // Initialize components
    Portfolio portfolio(1000000.0);
    RiskManager riskManager(0.1, 0.02, 0.15);
    BacktestEngine backtester(1000000.0);
    
    // Create strategies
    MeanReversionStrategy meanRevStrategy(&portfolio, &riskManager, 20, 2.0, 0.5);
    MomentumStrategy momentumStrategy(&portfolio, &riskManager, 12, 26);
    
    backtester.addStrategy(&meanRevStrategy);
    backtester.addStrategy(&momentumStrategy);
    
    // Load sample data
    std::vector<std::string> symbols = {"AAPL", "GOOGL", "MSFT"};
    for (const auto& symbol : symbols) {
        auto data = generateSampleData(symbol, 500);
        backtester.loadHistoricalData(symbol, data);
    }
    
    // Run backtest
    auto startDate = std::chrono::system_clock::now() - std::chrono::hours(24 * 400);
    auto endDate = std::chrono::system_clock::now() - std::chrono::hours(24 * 50);
    
    std::cout << "\nRunning backtest..." << std::endl;
    auto results = backtester.runBacktest(startDate, endDate);
    backtester.printResults(results);
    
    // Portfolio optimization example
    std::cout << "\n=== PORTFOLIO OPTIMIZATION ===" << std::endl;
    PortfolioOptimizer optimizer;
    
    // Add assets with sample return data
    for (const auto& symbol : symbols) {
        auto data = generateSampleData(symbol, 252);
        std::vector<double> returns;
        
        for (size_t i = 1; i < data.size(); ++i) {
            returns.push_back(std::log(data[i].close / data[i-1].close));
        }
        
        optimizer.addAsset(symbol, returns);
    }
    
    optimizer.calculateStatistics();
    auto optimalWeights = optimizer.optimizePortfolio(0.12); // Target 12% return
    optimizer.printOptimizationResults(optimalWeights);
    
    // Option pricing examples
    std::cout << "\n=== OPTION PRICING MODELS ===" << std::endl;
    
    OptionPricingModels::BSParams bsParams;
    bsParams.S = 150.0;  // Current stock price
    bsParams.K = 155.0;  // Strike price
    bsParams.T = 0.25;   // 3 months to expiration
    bsParams.r = 0.05;   // 5% risk-free rate
    bsParams.sigma = 0.25; // 25% volatility
    bsParams.q = 0.02;   // 2% dividend yield
    
    double bsCall = OptionPricingModels::blackScholesCall(bsParams);
    double bsPut = OptionPricingModels::blackScholesPut(bsParams);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Black-Scholes Call Price: $" << bsCall << std::endl;
    std::cout << "Black-Scholes Put Price:  $" << bsPut << std::endl;
    
    // Heston model pricing
    OptionPricingModels::HestonParams hestonParams;
    hestonParams.S = bsParams.S;
    hestonParams.K = bsParams.K;
    hestonParams.T = bsParams.T;
    hestonParams.r = bsParams.r;
    hestonParams.q = bsParams.q;
    hestonParams.v0 = 0.0625;     // Initial variance (25%^2)
    hestonParams.kappa = 2.0;     // Mean reversion speed
    hestonParams.theta = 0.0625;  // Long-term variance
    hestonParams.sigma = 0.3;     // Volatility of volatility
    hestonParams.rho = -0.7;      // Correlation
    
    std::cout << "\nCalculating Heston model price (Monte Carlo)..." << std::endl;
    double hestonCall = OptionPricingModels::hestonMonteCarlo(hestonParams, true, 50000, 100);
    std::cout << "Heston Call Price: $" << hestonCall << std::endl;
    
    // Binomial tree pricing
    double binomialCall = OptionPricingModels::binomialTree(
        bsParams.S, bsParams.K, bsParams.T, bsParams.r, bsParams.sigma, true, false, 200);
    double americanCall = OptionPricingModels::binomialTree(
        bsParams.S, bsParams.K, bsParams.T, bsParams.r, bsParams.sigma, true, true, 200);
    
    std::cout << "Binomial European Call: $" << binomialCall << std::endl;
    std::cout << "Binomial American Call: $" << americanCall << std::endl;
    
    // Risk analysis
    std::cout << "\n=== RISK ANALYSIS ===" << std::endl;
    
    double var95 = riskManager.calculateVaR(portfolio, 0.05, 1);
    std::cout << "Portfolio VaR (95%, 1-day): $" << std::abs(var95) << std::endl;
    
    // Stress testing
    RiskManager::StressScenario marketCrash;
    marketCrash.name = "Market Crash";
    marketCrash.shocks["AAPL"] = -0.30;   // 30% drop
    marketCrash.shocks["GOOGL"] = -0.25;  // 25% drop
    marketCrash.shocks["MSFT"] = -0.20;   // 20% drop
    
    double stressResult = riskManager.runStressTest(portfolio, marketCrash);
    std::cout << "Market Crash Scenario P&L: " << stressResult * 100 << "%" << std::endl;
    
    // Monte Carlo analysis
    std::cout << "\n=== MONTE CARLO ROBUSTNESS ANALYSIS ===" << std::endl;
    auto mcResults = backtester.monteCarloAnalysis(500);
    
    std::vector<double> sharpeRatios;
    std::vector<double> maxDrawdowns;
    
    for (const auto& result : mcResults) {
        sharpeRatios.push_back(result.sharpeRatio);
        maxDrawdowns.push_back(result.maxDrawdown);
    }
    
    std::sort(sharpeRatios.begin(), sharpeRatios.end());
    std::sort(maxDrawdowns.begin(), maxDrawdowns.end());
    
    std::cout << "Monte Carlo Results (500 simulations):" << std::endl;
    std::cout << "Sharpe Ratio - 5th percentile: " << sharpeRatios[25] << std::endl;
    std::cout << "Sharpe Ratio - 95th percentile: " << sharpeRatios[475] << std::endl;
    std::cout << "Max Drawdown - 5th percentile: " << maxDrawdowns[25] * 100 << "%" << std::endl;
    std::cout << "Max Drawdown - 95th percentile: " << maxDrawdowns[475] * 100 << "%" << std::endl;
    
    std::cout << "\n=== SYSTEM COMPLETE ===" << std::endl;
    
    return 0;
}