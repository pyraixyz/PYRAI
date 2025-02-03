module.exports = {
    skipFiles: ['mocks/', 'test/'],
    providerOptions: {
        default_balance_ether: 100000000,
        total_accounts: 20,
    },
    mocha: {
        enableTimeouts: false,
        before_timeout: 0
    },
    configureYulOptimizer: true,
    measureStatementCoverage: true,
    measureFunctionCoverage: true,
    measureBranchCoverage: true,
    measureModifierCoverage: true,
}; 