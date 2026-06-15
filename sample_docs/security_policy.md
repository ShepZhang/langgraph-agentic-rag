# Northstar Labs Security And Access Policy

Northstar Labs is a fictional company used for this sample knowledge base. This
policy contains no live credentials, infrastructure details, or sensitive
company information.

## Multi-Factor Authentication

Multi-factor authentication (MFA) is required for company email, source control,
cloud administration, production systems, the managed secrets vault, and any
other service that supports confidential company or customer information.

## Production Access

Production access is granted just in time (JIT) and expires after four hours.
Every request must include a ticket reference and approval from the on-call
engineering lead before access is activated.

Users must access production only for the approved purpose and must not share
temporary access with another person.

## Secrets Management

Secrets must be stored in the managed secrets vault. Passwords, API keys,
tokens, certificates, and other credentials must never be stored in source
control or shared documents.

## Access Reviews And Removal

Privileged access is reviewed quarterly. Access that is no longer required
must be removed within one business day after a role change, termination, or
review finding establishes that the access is unnecessary.

## Security Incident Reporting

Suspected credential exposure or unauthorized access must be reported to the
security incident channel within 30 minutes of discovery. The report should
include the affected account or system, the time discovered, and any immediate
containment already performed, but must not include exposed secret values.
